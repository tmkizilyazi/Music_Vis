use std::ffi::CString;

pub struct ShaderProgram {
    id: u32,
}

impl ShaderProgram {
    pub fn new(vertex_source: &str, fragment_source: &str) -> Result<Self, String> {
        let vertex_shader = compile_shader(vertex_source, gl::VERTEX_SHADER)?;
        let fragment_shader = compile_shader(fragment_source, gl::FRAGMENT_SHADER)?;

        unsafe {
            let program = gl::CreateProgram();
            gl::AttachShader(program, vertex_shader);
            gl::AttachShader(program, fragment_shader);
            gl::LinkProgram(program);

            // Check for linking errors
            let mut success = 0;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
            if success == 0 {
                let mut len = 0;
                gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
                let mut info_log = Vec::with_capacity(len as usize);
                gl::GetProgramInfoLog(
                    program,
                    len,
                    std::ptr::null_mut(),
                    info_log.as_mut_ptr() as *mut i8,
                );
                info_log.set_len(len as usize);
                return Err(String::from_utf8_lossy(&info_log).to_string());
            }

            gl::DeleteShader(vertex_shader);
            gl::DeleteShader(fragment_shader);

            Ok(ShaderProgram { id: program })
        }
    }

    pub fn use_program(&self) {
        unsafe {
            gl::UseProgram(self.id);
        }
    }

    pub fn set_mat4(&self, name: &str, value: &nalgebra_glm::Mat4) {
        unsafe {
            let name = CString::new(name).unwrap();
            let location = gl::GetUniformLocation(self.id, name.as_ptr());
            gl::UniformMatrix4fv(location, 1, gl::FALSE, value.as_ptr());
        }
    }

    pub fn set_vec4(&self, name: &str, value: &nalgebra_glm::Vec4) {
        unsafe {
            let name = CString::new(name).unwrap();
            let location = gl::GetUniformLocation(self.id, name.as_ptr());
            gl::Uniform4fv(location, 1, value.as_ptr());
        }
    }

    pub fn set_float(&self, name: &str, value: f32) {
        unsafe {
            let name = CString::new(name).unwrap();
            let location = gl::GetUniformLocation(self.id, name.as_ptr());
            gl::Uniform1f(location, value);
        }
    }
}

fn compile_shader(source: &str, shader_type: u32) -> Result<u32, String> {
    unsafe {
        let shader = gl::CreateShader(shader_type);
        let c_str = CString::new(source.as_bytes()).unwrap();
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), std::ptr::null());
        gl::CompileShader(shader);

        let mut success = 0;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);
        if success == 0 {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut info_log = Vec::with_capacity(len as usize);
            gl::GetShaderInfoLog(
                shader,
                len,
                std::ptr::null_mut(),
                info_log.as_mut_ptr() as *mut i8,
            );
            info_log.set_len(len as usize);
            return Err(String::from_utf8_lossy(&info_log).to_string());
        }

        Ok(shader)
    }
}

pub const VERTEX_SHADER: &str = r#"
    #version 330 core
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform float time;
    uniform float audioEnergy;
    uniform float bassEnergy;
    uniform float midEnergy;
    uniform float highEnergy;
    
    out vec3 FragPos;
    out vec2 TexCoord;
    out float Energy;
    out vec3 Normal;
    out float VertexGlow;
    
    // Dalga fonksiyonu
    float wave(vec3 pos, float freq, float amp) {
        return sin(pos.x * freq + time) * cos(pos.z * freq + time) * amp;
    }
    
    void main() {
        vec3 pos = aPos;
        
        // Çoklu dalga deformasyonu
        float baseFreq = 2.0 + bassEnergy * 3.0;
        float wave1 = wave(pos, baseFreq, 0.3 * midEnergy);
        float wave2 = wave(pos * 1.5, baseFreq * 2.0, 0.2 * highEnergy);
        float wave3 = wave(pos * 0.5, baseFreq * 0.5, 0.4 * bassEnergy);
        
        pos += pos * (wave1 + wave2 + wave3);
        
        // Spiral hareket
        float spiral = length(pos.xz) * 2.0;
        float spiralIntensity = sin(spiral + time * 2.0) * (bassEnergy + 0.3);
        pos.y += spiralIntensity;
        
        // Dönme ve büzülme
        float twist = time * 0.5 + highEnergy * 2.0;
        float angle = twist + spiral;
        mat2 rotation = mat2(
            cos(angle), -sin(angle),
            sin(angle), cos(angle)
        );
        pos.xz = rotation * pos.xz;
        
        // Nabız efekti
        float pulse = sin(time * (2.0 + bassEnergy * 3.0)) * 0.5 + 0.5;
        pos *= 1.0 + pulse * audioEnergy * 0.3;
        
        // Vertex parlaklığı
        VertexGlow = pulse * (1.0 - length(pos) * 0.5) + highEnergy * 0.5;
        
        FragPos = vec3(model * vec4(pos, 1.0));
        TexCoord = pos.xy * 0.5 + 0.5;
        Energy = audioEnergy;
        Normal = normalize(pos);
        
        gl_Position = projection * view * model * vec4(pos, 1.0);
    }
"#;

pub const FRAGMENT_SHADER: &str = r#"
    #version 330 core
    out vec4 FragColor;
    
    in vec3 FragPos;
    in vec2 TexCoord;
    in float Energy;
    in vec3 Normal;
    in float VertexGlow;
    
    uniform vec4 color;
    uniform float time;
    uniform float bassEnergy;
    uniform float midEnergy;
    uniform float highEnergy;
    
    // Kaleidoskop efekti
    vec2 kaleidoscope(vec2 uv, float segments) {
        float angle = atan(uv.y, uv.x);
        float radius = length(uv);
        angle = mod(angle, 3.14159 * 2.0 / segments) - 3.14159 / segments;
        return vec2(cos(angle), sin(angle)) * radius;
    }
    
    // Fraktal noise
    float noise(vec2 p) {
        return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }
    
    // Rainbow renk
    vec3 rainbow(float t) {
        vec3 c = 0.5 + 0.5 * cos(6.28318 * (t + vec3(0.0, 0.33, 0.67)));
        return mix(c, vec3(1.0), 0.2);
    }
    
    void main() {
        vec2 uv = TexCoord * 2.0 - 1.0;
        vec3 finalColor = color.rgb;
        
        // Zaman bazlı renk kayması
        float timeShift = time * 0.5;
        vec3 shiftedColor = rainbow(timeShift + length(uv) * 0.2);
        
        // Kaleidoskop efekti
        float segments = 8.0 + sin(time + bassEnergy * 5.0) * 4.0;
        vec2 kaleid = kaleidoscope(uv, segments);
        
        // Spiral dalgalar
        float spiral = atan(kaleid.y, kaleid.x) / 6.28318 + 0.5;
        float rings = length(kaleid) * 10.0 + time * 2.0;
        float waves = sin(rings + spiral * 20.0) * 0.5 + 0.5;
        
        // Fraktal doku
        float zoom = 5.0 + sin(time) * 2.0;
        vec2 fractalUV = kaleid * zoom;
        float fractal = 0.0;
        float amp = 0.5;
        for(int i = 0; i < 5; i++) {
            fractal += noise(fractalUV) * amp;
            fractalUV *= 2.0;
            fractalUV = kaleidoscope(fractalUV, 4.0 + float(i));
            amp *= 0.5;
        }
        
        // Neon parlaması
        vec3 neonColor = rainbow(timeShift * 0.7) * (bassEnergy + 0.5);
        float neonGlow = pow(waves * fractal, 2.0) * (midEnergy + 0.5);
        
        // Renk katmanları
        finalColor = mix(finalColor, shiftedColor, 0.6);
        finalColor += neonColor * neonGlow * 0.5;
        finalColor += rainbow(fractal + timeShift) * highEnergy * 0.3;
        
        // Kenar efektleri
        float edge = pow(1.0 - abs(dot(Normal, vec3(0.0, 0.0, 1.0))), 2.0);
        finalColor += rainbow(edge + timeShift) * edge * (bassEnergy + 0.2);
        
        // Glitch efekti
        float glitchIntensity = step(0.98, sin(time * 50.0)) * highEnergy;
        vec3 glitchColor = rainbow(noise(uv * 100.0 + time));
        finalColor = mix(finalColor, glitchColor, glitchIntensity * 0.5);
        
        // Renk doygunluğu artırma
        finalColor = pow(finalColor, vec3(0.8)); // Renkleri daha canlı yap
        finalColor *= 1.2; // Parlaklığı artır
        
        // HDR ve ton eşleme
        finalColor = finalColor / (finalColor + vec3(1.0));
        finalColor = pow(finalColor, vec3(1.0 / 2.2));
        
        // Alpha kanalı
        float alpha = color.a + edge * 0.5 + waves * 0.3;
        alpha = min(alpha, 1.0);
        
        FragColor = vec4(finalColor, alpha);
    }
"#;
