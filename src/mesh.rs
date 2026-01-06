use std::collections::HashMap;

use glam::Vec3;
use lazy_vulkan::{Allocator, BufferAllocation};

use crate::{
    LoadedMaterial, LoadedMesh, MaterialID, MeshID, Primitive, Vertex, get_primitive,
    load_primitive,
};

#[derive(Default, Debug, Clone)]
pub struct Mesh {
    pub id: MeshID,
    pub primitives: Vec<Primitive>,
}

pub fn get_mesh(mesh: gltf::Mesh, blob: &[u8]) -> Mesh {
    let id = MeshID(mesh.index() as _);
    let primitives = mesh
        .primitives()
        .filter_map(|p| get_primitive(p, blob))
        .collect::<Vec<_>>();

    Mesh { id, primitives }
}

pub fn load_mesh(
    mesh: &Mesh,
    allocator: &mut Allocator,
    vertex_buffer: &mut BufferAllocation<Vertex>,
    index_buffer: &mut BufferAllocation<u32>,
    loaded_materials: &HashMap<MaterialID, LoadedMaterial>,
) -> LoadedMesh {
    let primitives = mesh
        .primitives
        .iter()
        .map(|p| load_primitive(p, allocator, vertex_buffer, index_buffer, loaded_materials))
        .collect();

    LoadedMesh {
        id: mesh.id,
        primitives,
    }
}

/// Simple axis aligned bounding box in the mesh's coordinate space
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for AABB {
    fn default() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }
}

impl AABB {
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    pub fn expand_to_include_point(&mut self, point_in_local_space: Vec3) {
        self.min = self.min.min(point_in_local_space);
        self.max = self.max.max(point_in_local_space);
    }

    pub fn extend(&mut self, aabb: &AABB) {
        self.expand_to_include_point(aabb.min);
        self.expand_to_include_point(aabb.max);
    }
}
