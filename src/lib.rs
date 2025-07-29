use std::path::Path;

pub const NO_TEXTURE: u32 = u32::MAX;

#[derive(Default, Debug, Clone)]
pub struct MaterialID(u32);

impl From<usize> for MaterialID {
    fn from(value: usize) -> Self {
        MaterialID(value as u32)
    }
}

#[derive(Default, Debug, Clone)]
pub struct MeshID(u32);

impl From<usize> for MeshID {
    fn from(value: usize) -> Self {
        MeshID(value as u32)
    }
}

#[derive(Default, Debug, Clone)]
pub struct TextureID(u32);

impl From<usize> for TextureID {
    fn from(value: usize) -> Self {
        TextureID(value as u32)
    }
}

#[derive(Default, Debug, Clone)]
pub struct Asset {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

#[derive(Default, Debug, Clone)]
pub struct Mesh {
    pub id: MeshID,
    pub primitives: Vec<Primitive>,
}

#[derive(Default, Debug, Clone)]
pub struct Primitive {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub material: Option<MaterialID>,
}

#[derive(Default, Debug, Clone)]
pub struct Vertex {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub uv: glam::Vec2,
}

impl Vertex {
    pub fn new(
        position: impl Into<glam::Vec3>,
        normal: impl Into<glam::Vec3>,
        uv: Option<impl Default + Into<glam::Vec2>>,
    ) -> Vertex {
        Vertex {
            position: position.into(),
            normal: normal.into(),
            uv: uv.unwrap_or_default().into(),
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Material {
    id: MaterialID,
    base_colour_factor: glam::Vec4,
    base_colour_texture: TextureID,
    normal_texture: TextureID,
    metallic_roughness_texture: TextureID,
    ao_texture: TextureID, // in the common case this is just the red channel of the MR texture
}

pub fn load_asset(path: impl AsRef<Path>) -> Result<Asset, gltf::Error> {
    let gltf::Gltf { document, blob } = gltf::Gltf::open(path)?;
    let Some(blob) = blob else {
        // We only support GLB files
        return Err(gltf::Error::MissingBlob);
    };

    let meshes = document.meshes().map(|m| load_mesh(m, &blob)).collect();
    let materials = document
        .materials()
        .filter_map(|m| load_material(m))
        .collect();

    Ok(Asset { meshes, materials })
}

fn load_mesh(mesh: gltf::Mesh, blob: &[u8]) -> Mesh {
    Mesh {
        id: MeshID(mesh.index() as _),
        primitives: mesh
            .primitives()
            .filter_map(|p| load_primitive(p, blob))
            .collect(),
    }
}

fn load_material(material: gltf::Material) -> Option<Material> {
    // We don't want the default material
    let Some(id) = material.index() else {
        return None;
    };

    let pbr = material.pbr_metallic_roughness();
    let base_colour_texture = pbr
        .base_color_texture()
        .map(|t| t.texture().index())
        .unwrap_or(NO_TEXTURE as _)
        .into();
    let metallic_roughness_texture = pbr
        .metallic_roughness_texture()
        .map(|t| t.texture().index())
        .unwrap_or(NO_TEXTURE as _)
        .into();
    let normal_texture = material
        .normal_texture()
        .map(|t| t.texture().index())
        .unwrap_or(NO_TEXTURE as _)
        .into();
    let ao_texture = material
        .occlusion_texture()
        .map(|t| t.texture().index())
        .unwrap_or(NO_TEXTURE as _)
        .into();

    Some(Material {
        id: id.into(),
        base_colour_factor: pbr.base_color_factor().into(),
        base_colour_texture,
        metallic_roughness_texture,
        normal_texture,
        ao_texture,
    })
}

fn load_primitive(primitive: gltf::Primitive, blob: &[u8]) -> Option<Primitive> {
    let reader = primitive.reader(|_| Some(blob));

    let indices = reader.read_indices()?.into_u32().collect();

    // This is hilarious! Thanks, ChatGPT
    // - UV coordinates are optional for us
    // - in the case that there is no TEXCOORD1 attribute, the iterator will return None each time
    // - otherwise the iterator will yield the next coordinate
    let mut uv_iter = reader
        .read_tex_coords(0)
        .into_iter()
        .flat_map(|tc| tc.into_f32());

    // Since we require positions and normals, we can just zip through those iterators and build
    // our vertex from there.
    let vertices = reader
        .read_positions()?
        .zip(reader.read_normals()?)
        .map(|(position, normal)| Vertex::new(position, normal, uv_iter.next()))
        .collect();

    // Materials are optional, so grab the ID if it exists
    let material = primitive.material().index().map(|i| MaterialID(i as _));

    Some(Primitive {
        vertices,
        indices,
        material,
    })
}

#[cfg(test)]
mod tests {
    use super::load_asset;

    #[test]
    fn it_works() {
        let asset = load_asset("test_assets/cube.glb").unwrap();
        assert_eq!(asset.meshes.len(), 1);
        assert_eq!(asset.meshes[0].primitives.len(), 1);
        assert_eq!(asset.meshes[0].primitives[0].indices.len(), 2916);
        assert_eq!(asset.meshes[0].primitives[0].vertices.len(), 681);
    }
}
