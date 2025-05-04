mod shaders;

use glfw::{Action, Context, Key};
use nalgebra_glm as glm;
use rand::Rng;
use rodio::{Decoder, OutputStream, Source};
use rustfft::{num_complex::Complex, FftPlanner};
use std::fs::File;
use std::io::BufReader;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::shaders::{ShaderProgram, FRAGMENT_SHADER, VERTEX_SHADER};

const SAMPLE_RATE: u32 = 44100;
const FFT_SIZE: usize = 2048;
const MIN_DB: f32 = -60.0;
const MAX_DB: f32 = 0.0;

struct AudioAnalyzer {
    spectrum: Arc<Mutex<Vec<f32>>>,
    bass_energy: Arc<Mutex<f32>>,
    mid_energy: Arc<Mutex<f32>>,
    high_energy: Arc<Mutex<f32>>,
    _stream: Option<OutputStream>,
}

impl AudioAnalyzer {
    fn new() -> Self {
        Self {
            spectrum: Arc::new(Mutex::new(vec![0.0; FFT_SIZE / 2])),
            bass_energy: Arc::new(Mutex::new(0.0)),
            mid_energy: Arc::new(Mutex::new(0.0)),
            high_energy: Arc::new(Mutex::new(0.0)),
            _stream: None,
        }
    }

    fn start_audio_processing(&mut self, file_path: &str) {
        let (stream, stream_handle) = OutputStream::try_default().unwrap();

        // Müzik çalma için
        let file_play = BufReader::new(File::open(file_path).unwrap());
        let source_play = Decoder::new(file_play).unwrap();
        let _ = stream_handle.play_raw(source_play.convert_samples());

        // FFT analizi için
        let file_analyze = BufReader::new(File::open(file_path).unwrap());
        let source_analyze = Decoder::new(file_analyze).unwrap();
        let samples: Vec<f32> = source_analyze.convert_samples().collect();

        self._stream = Some(stream);

        let spectrum = self.spectrum.clone();
        let bass = self.bass_energy.clone();
        let mid = self.mid_energy.clone();
        let high = self.high_energy.clone();

        thread::spawn(move || {
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(FFT_SIZE);
            let mut buffer = vec![Complex::new(0.0, 0.0); FFT_SIZE];
            let mut pos = 0;

            loop {
                for i in 0..FFT_SIZE {
                    if pos + i < samples.len() {
                        buffer[i] = Complex::new(samples[pos + i], 0.0);
                    } else {
                        buffer[i] = Complex::new(0.0, 0.0);
                    }
                }

                fft.process(&mut buffer);

                let mut spectrum_data = vec![0.0; FFT_SIZE / 2];
                for i in 0..FFT_SIZE / 2 {
                    let magnitude = (buffer[i].norm() / FFT_SIZE as f32).log10() * 20.0;
                    spectrum_data[i] = (magnitude - MIN_DB) / (MAX_DB - MIN_DB);
                }

                let mut bass_sum = 0.0;
                let mut mid_sum = 0.0;
                let mut high_sum = 0.0;

                for i in 0..FFT_SIZE / 2 {
                    let freq = i as f32 * SAMPLE_RATE as f32 / FFT_SIZE as f32;
                    if freq < 250.0 {
                        bass_sum += spectrum_data[i];
                    } else if freq < 2000.0 {
                        mid_sum += spectrum_data[i];
                    } else {
                        high_sum += spectrum_data[i];
                    }
                }

                *spectrum.lock().unwrap() = spectrum_data;
                *bass.lock().unwrap() = bass_sum / 250.0;
                *mid.lock().unwrap() = mid_sum / 1750.0;
                *high.lock().unwrap() = high_sum / (FFT_SIZE as f32 / 2.0 - 2000.0);

                pos += FFT_SIZE / 2;
                if pos >= samples.len() {
                    pos = 0;
                }

                thread::sleep(std::time::Duration::from_millis(16));
            }
        });
    }
}

struct Visualizer {
    shader_program: ShaderProgram,
    time: f32,
    audio_analyzer: Arc<AudioAnalyzer>,
    shapes: Vec<Shape>,
    vao: u32,
    vbo: u32,
}

struct Shape {
    position: glm::Vec3,
    scale: f32,
    color: glm::Vec4,
    rotation: f32,
    energy_response: f32,
}

impl Visualizer {
    fn new(audio_analyzer: Arc<AudioAnalyzer>) -> Self {
        let (vao, vbo) = unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

            // Küp köşe noktaları
            let vertices: [f32; 108] = [
                // Ön yüz
                -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5,
                -0.5, 0.5, // Arka yüz
                -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5,
                -0.5, -0.5, -0.5, // Üst yüz
                -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5,
                -0.5, 0.5, -0.5, // Alt yüz
                -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5,
                -0.5, -0.5, -0.5, // Sağ yüz
                0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
                -0.5, -0.5, // Sol yüz
                -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5,
                -0.5, -0.5, -0.5,
            ];

            let mut vao = 0;
            let mut vbo = 0;

            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vbo);

            gl::BindVertexArray(vao);
            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (vertices.len() * std::mem::size_of::<f32>()) as isize,
                vertices.as_ptr() as *const _,
                gl::STATIC_DRAW,
            );

            gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 0, std::ptr::null());
            gl::EnableVertexAttribArray(0);

            (vao, vbo)
        };

        let shader_program = ShaderProgram::new(VERTEX_SHADER, FRAGMENT_SHADER)
            .expect("Failed to create shader program");

        let mut shapes = Vec::new();
        let mut rng = rand::thread_rng();

        // İç içe tüneller oluştur
        for tunnel_id in 0..3 {
            let base_radius = 3.0 + tunnel_id as f32 * 4.0;

            // Her tünel için spiral şekiller
            for i in 0..120 {
                let ring_count = 12;
                let angle_step = std::f32::consts::PI * 2.0 / ring_count as f32;

                for j in 0..ring_count {
                    let angle = j as f32 * angle_step;
                    let z_pos = (i as f32 * 1.5) - 90.0;

                    // Spiral şekil
                    let spiral_factor = (i as f32 * 0.1).sin() * 2.0;
                    let radius = base_radius + spiral_factor;

                    // Alternatif şekiller için offset
                    let offset_x = (i as f32 * 0.2).sin() * 2.0;
                    let offset_y = (i as f32 * 0.15).cos() * 2.0;

                    shapes.push(Shape {
                        position: glm::vec3(
                            angle.cos() * radius + offset_x,
                            angle.sin() * radius + offset_y,
                            z_pos,
                        ),
                        scale: rng.gen_range(0.2..0.5),
                        color: glm::vec4(
                            rng.gen_range(0.6..1.0),
                            rng.gen_range(0.6..1.0),
                            rng.gen_range(0.6..1.0),
                            rng.gen_range(0.6..0.9),
                        ),
                        rotation: angle + (tunnel_id as f32 * std::f32::consts::PI / 3.0),
                        energy_response: rng.gen_range(0.8..2.0),
                    });

                    // İç şekiller ekle
                    if rng.gen_bool(0.3) {
                        let inner_radius = radius * 0.5;
                        let inner_z = z_pos + rng.gen_range(-1.0..1.0);

                        shapes.push(Shape {
                            position: glm::vec3(
                                angle.cos() * inner_radius,
                                angle.sin() * inner_radius,
                                inner_z,
                            ),
                            scale: rng.gen_range(0.1..0.3),
                            color: glm::vec4(
                                rng.gen_range(0.7..1.0),
                                rng.gen_range(0.7..1.0),
                                rng.gen_range(0.7..1.0),
                                rng.gen_range(0.7..1.0),
                            ),
                            rotation: -angle * 2.0,
                            energy_response: rng.gen_range(1.0..2.5),
                        });
                    }
                }
            }
        }

        Self {
            shader_program,
            time: 0.0,
            audio_analyzer,
            shapes,
            vao,
            vbo,
        }
    }

    fn render(&mut self) {
        self.time += 0.016;

        let bass = *self.audio_analyzer.bass_energy.lock().unwrap();
        let mid = *self.audio_analyzer.mid_energy.lock().unwrap();
        let high = *self.audio_analyzer.high_energy.lock().unwrap();

        unsafe {
            gl::ClearColor(0.0, 0.0, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            // Kamera hareketi
            let forward_speed = 1.5 + bass * 2.0;
            let camera_z = -50.0 + self.time * forward_speed;
            let camera_y = 2.0 + (self.time * 0.3).sin() * 2.0;
            let camera_x = (self.time * 0.2).cos() * 4.0;

            let target_z = camera_z + 10.0;
            let target_y = camera_y + (mid * 2.0).sin() * 3.0;
            let target_x = camera_x + (high * 2.0).cos() * 3.0;

            let up_vector = glm::vec3(
                (self.time * 0.1).sin() * 0.2,
                1.0,
                (self.time * 0.1).cos() * 0.2,
            );

            let view = glm::look_at(
                &glm::vec3(camera_x, camera_y, camera_z),
                &glm::vec3(target_x, target_y, target_z),
                &up_vector,
            );

            let projection = glm::perspective(70.0f32.to_radians(), 800.0 / 600.0, 0.1, 100.0);

            self.shader_program.use_program();
            self.shader_program.set_mat4("view", &view);
            self.shader_program.set_mat4("projection", &projection);
            self.shader_program.set_float("time", self.time);
            self.shader_program.set_float("bassEnergy", bass);
            self.shader_program.set_float("midEnergy", mid);
            self.shader_program.set_float("highEnergy", high);

            for shape in &mut self.shapes {
                let mut model = glm::Mat4::identity();

                let mut pos = shape.position;
                pos.z = pos.z + camera_z + 100.0;
                if pos.z > camera_z + 10.0 {
                    pos.z -= 180.0;
                }

                let energy = bass * shape.energy_response;
                let scale = shape.scale * (1.0 + energy);

                model = glm::translate(&model, &pos);
                model = glm::rotate(
                    &model,
                    self.time * 0.5 + shape.rotation,
                    &glm::vec3(0.0, 1.0, 0.0),
                );
                model = glm::scale(&model, &glm::vec3(scale, scale, scale));

                let color = glm::vec4(
                    shape.color.x + mid * 0.3 * (self.time * 1.5 + pos.x).sin(),
                    shape.color.y + high * 0.3 * (self.time * 2.0 + pos.y).sin(),
                    shape.color.z + bass * 0.3 * (self.time * 1.0 + pos.z).sin(),
                    shape.color.w,
                );

                self.shader_program.set_mat4("model", &model);
                self.shader_program.set_vec4("color", &color);
                self.shader_program.set_float("audioEnergy", energy);

                gl::DrawArrays(gl::TRIANGLES, 0, 36);
            }
        }
    }
}

impl Drop for Visualizer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteVertexArrays(1, &self.vao);
            gl::DeleteBuffers(1, &self.vbo);
        }
    }
}

fn main() {
    let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(
        glfw::OpenGlProfileHint::Core,
    ));

    let (mut window, events) = glfw
        .create_window(
            800,
            600,
            "Berlin Techno Visualizer",
            glfw::WindowMode::Windowed,
        )
        .expect("Failed to create GLFW window");

    window.make_current();
    window.set_key_polling(true);

    gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);

    let mut audio_analyzer = Arc::new(AudioAnalyzer::new());
    Arc::get_mut(&mut audio_analyzer)
        .unwrap()
        .start_audio_processing(
            "src/Daft Punk - Veridis Quo (Official Video) (online-audio-converter.com).mp3",
        );

    let mut visualizer = Visualizer::new(audio_analyzer);

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            match event {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                    window.set_should_close(true)
                }
                _ => {}
            }
        }

        visualizer.render();
        window.swap_buffers();
    }
}
