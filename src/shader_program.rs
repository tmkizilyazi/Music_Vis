use gl::types::*;
use std::ffi::CString;
use std::ptr;

pub struct ShaderProgram {
    id: GLuint,
}

impl ShaderProgram {
    pub fn new(vertex_source: &str, fragment_source: &str) -> Result<Self, String> {
        let program = unsafe {
            let vertex_shader = gl::CreateShader(gl::VERTEX_SHADER);
            let c_str_vert = CString::new(vertex_source.as_bytes()).unwrap();
            gl::ShaderSource(vertex_shader, 1, &c_str_vert.as_ptr(), ptr::null());
            gl::CompileShader(vertex_shader);

            let mut success = gl::FALSE as GLint;
            let mut info_log = Vec::with_capacity(512);
            info_log.set_len(512 - 1);
            gl::GetShaderiv(vertex_shader, gl::COMPILE_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                gl::GetShaderInfoLog(
                    vertex_shader,
                    512,
                    ptr::null_mut(),
                    info_log.as_mut_ptr() as *mut GLchar,
                );
                return Err(String::from_utf8_lossy(&info_log).to_string());
            }

            let fragment_shader = gl::CreateShader(gl::FRAGMENT_SHADER);
            let c_str_frag = CString::new(fragment_source.as_bytes()).unwrap();
            gl::ShaderSource(fragment_shader, 1, &c_str_frag.as_ptr(), ptr::null());
            gl::CompileShader(fragment_shader);

            gl::GetShaderiv(fragment_shader, gl::COMPILE_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                gl::GetShaderInfoLog(
                    fragment_shader,
                    512,
                    ptr::null_mut(),
                    info_log.as_mut_ptr() as *mut GLchar,
                );
                return Err(String::from_utf8_lossy(&info_log).to_string());
            }

            let program = gl::CreateProgram();
            gl::AttachShader(program, vertex_shader);
            gl::AttachShader(program, fragment_shader);
            gl::LinkProgram(program);

            gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);
            if success != gl::TRUE as GLint {
                gl::GetProgramInfoLog(
                    program,
                    512,
                    ptr::null_mut(),
                    info_log.as_mut_ptr() as *mut GLchar,
                );
                return Err(String::from_utf8_lossy(&info_log).to_string());
            }

            gl::DeleteShader(vertex_shader);
            gl::DeleteShader(fragment_shader);

            program
        };

        Ok(ShaderProgram { id: program })
    }

    pub fn use_program(&self) {
        unsafe {
            gl::UseProgram(self.id);
        }
    }

    pub fn set_mat4(&self, name: &str, value: &nalgebra_glm::Mat4) {
        unsafe {
            let c_name = CString::new(name).unwrap();
            let location = gl::GetUniformLocation(self.id, c_name.as_ptr());
            gl::UniformMatrix4fv(location, 1, gl::FALSE, value.as_ptr());
        }
    }

    pub fn set_vec3(&self, name: &str, value: &nalgebra_glm::Vec3) {
        unsafe {
            let c_name = CString::new(name).unwrap();
            let location = gl::GetUniformLocation(self.id, c_name.as_ptr());
            gl::Uniform3fv(location, 1, value.as_ptr());
        }
    }

    pub fn set_vec4(&self, name: &str, value: &nalgebra_glm::Vec4) {
        unsafe {
            let c_name = CString::new(name).unwrap();
            let location = gl::GetUniformLocation(self.id, c_name.as_ptr());
            gl::Uniform4fv(location, 1, value.as_ptr());
        }
    }

    pub fn set_float(&self, name: &str, value: f32) {
        unsafe {
            let c_name = CString::new(name).unwrap();
            let location = gl::GetUniformLocation(self.id, c_name.as_ptr());
            gl::Uniform1f(location, value);
        }
    }
}

impl Drop for ShaderProgram {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.id);
        }
    }
}
