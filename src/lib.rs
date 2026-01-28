mod mesh;

use std::{
    collections::HashMap,
    fmt::Display,
    path::Path,
    sync::atomic::{AtomicU32, Ordering},
};

pub use mesh::{AABB, Mesh, get_mesh, load_mesh};

use lazy_vulkan::{Allocator, BufferAllocation, Image, ImageManager, SlabUpload, ash::vk};

pub const NO_TEXTURE: u32 = u32::MAX;

#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Copy, Hash)]
pub struct MaterialID(u32);

impl From<usize> for MaterialID {
    fn from(value: usize) -> Self {
        MaterialID(value as u32)
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub struct MeshID(u32);

impl From<usize> for MeshID {
    fn from(value: usize) -> Self {
        MeshID(value as u32)
    }
}

impl From<MeshID> for usize {
    fn from(value: MeshID) -> Self {
        value.0 as usize
    }
}

impl Display for MeshID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub struct NodeID(u32);

impl From<usize> for NodeID {
    fn from(value: usize) -> Self {
        NodeID(value as u32)
    }
}

impl From<NodeID> for usize {
    fn from(value: NodeID) -> Self {
        value.0 as usize
    }
}

impl Display for NodeID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Copy, Hash, PartialOrd, Ord)]
pub struct PrimitiveID(u32);
impl PrimitiveID {
    fn next() -> PrimitiveID {
        PrimitiveID(NEXT_PRIMITIVE_ID.fetch_add(1, Ordering::Relaxed))
    }
}

static NEXT_PRIMITIVE_ID: AtomicU32 = AtomicU32::new(0);

impl From<usize> for PrimitiveID {
    fn from(value: usize) -> Self {
        PrimitiveID(value as u32)
    }
}

impl From<PrimitiveID> for u32 {
    fn from(value: PrimitiveID) -> Self {
        value.0
    }
}

impl Display for PrimitiveID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Copy, Hash)]
pub struct TextureID(u32);

impl From<usize> for TextureID {
    fn from(value: usize) -> Self {
        TextureID(value as u32)
    }
}

impl PartialEq<usize> for TextureID {
    fn eq(&self, other: &usize) -> bool {
        *other as u32 == self.0
    }
}

impl PartialEq<TextureID> for usize {
    fn eq(&self, other: &TextureID) -> bool {
        *other == TextureID::from(*self)
    }
}

impl From<u32> for TextureID {
    fn from(value: u32) -> Self {
        TextureID(value)
    }
}

#[derive(Default, Clone, Debug)]
pub struct LoadedAsset {
    pub meshes: Vec<LoadedMesh>,
    pub nodes: Vec<Node>,
    pub aabb: AABB,
}

#[derive(Default, Clone, Debug)]
pub struct LoadedMesh {
    pub id: MeshID,
    pub primitives: Vec<LoadedPrimitive>,
}

#[derive(Clone)]
pub struct LoadedMaterial {
    pub id: MaterialID,
    pub material: SlabUpload<GPUMaterial>,
}

#[derive(Debug, Clone)]
pub struct LoadedTexture {
    pub id: TextureID,
    pub image: Image,
}

impl From<Image> for LoadedTexture {
    fn from(image: Image) -> Self {
        let id = image.id.into();
        LoadedTexture { id, image }
    }
}

#[derive(Clone, Debug)]
pub struct LoadedPrimitive {
    pub id: PrimitiveID,
    pub index_count: u32,
    pub vertex_count: u32,
    pub index_buffer_offset: u64,
    pub vertex_buffer_offset: u64,
    pub material: vk::DeviceAddress,
}

#[derive(Default, Debug, Clone)]
pub struct Asset {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub textures: Vec<Texture>,
    pub nodes: Vec<Node>,
}

#[derive(Default, Debug, Clone)]
pub struct Node {
    pub id: NodeID,
    pub mesh_id: MeshID,
    pub children: Vec<NodeID>,
    pub transform: glam::Affine3A,
}

#[derive(Default, Debug, Clone)]
pub struct Primitive {
    pub id: PrimitiveID,
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
    pub material: MaterialID,
}

#[derive(Default, Debug, Clone)]
pub struct Texture {
    pub id: TextureID,
    // Decoded into [`self.format`]
    pub image_bytes: Vec<u8>,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
}

#[derive(Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    pub position: glam::Vec3,
    pub normal: glam::Vec3,
    pub uv: glam::Vec2,
}

unsafe impl bytemuck::Zeroable for Vertex {}
unsafe impl bytemuck::Pod for Vertex {}

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

#[derive(Default, Debug, Clone, Copy)]
pub struct Material {
    pub id: MaterialID,
    pub base_colour_factor: glam::Vec4,
    pub emissive_colour_factor: glam::Vec3,
    pub base_colour_texture: TextureID,
    pub normal_texture: TextureID,
    pub metallic_roughness_texture: TextureID,
    pub ao_texture: TextureID, // in the common case this is just the red channel of the MR texture
}

#[derive(Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct GPUMaterial {
    pub base_colour_factor: glam::Vec4,
    pub emissive_colour_factor: glam::Vec3,
    pub base_colour_texture: TextureID,
    pub normal_texture: TextureID,
    pub metallic_roughness_texture: TextureID,
    pub ao_texture: TextureID, // in the common case this is just the red channel of the MR texture
}

unsafe impl bytemuck::Zeroable for GPUMaterial {}
unsafe impl bytemuck::Pod for GPUMaterial {}

pub fn load_asset(
    path: impl AsRef<Path>,
    allocator: &mut Allocator,
    image_manager: &mut ImageManager,
    index_buffer: &mut BufferAllocation<u32>,
    vertex_buffer: &mut BufferAllocation<Vertex>,
) -> Result<LoadedAsset, gltf::Error> {
    let asset = get_asset(path)?;

    // First, load in all our textures
    let loaded_textures = asset
        .textures
        .iter()
        .map(|t| load_texture(t, allocator, image_manager))
        .collect();

    // Then, patch up the materials to point to the loaded textures and load them in
    let loaded_materials = asset
        .materials
        .iter()
        .map(|m| load_material(m, allocator, &loaded_textures))
        .collect();

    // Next, patch up the meshes to point to the loaded materials and load them in
    let meshes = asset
        .meshes
        .iter()
        .map(|m| load_mesh(m, allocator, vertex_buffer, index_buffer, &loaded_materials))
        .collect();

    // Finally, calculate an AABB for this asset by walking through each node, transforming each
    // position into the mesh's coordinate space and expanding the AABB accordingly.
    let mut asset_aabb = AABB::default();
    for node in &asset.nodes {
        for primitive in &asset.meshes[usize::from(node.mesh_id)].primitives {
            for &vertex in &primitive.vertices {
                let point_in_local_space = node.transform.transform_point3(vertex.position);
                asset_aabb.expand_to_include_point(point_in_local_space);
            }
        }
    }

    Ok(LoadedAsset {
        meshes,
        nodes: asset.nodes,
        aabb: asset_aabb,
    })
}

pub fn get_asset(path: impl AsRef<Path>) -> Result<Asset, gltf::Error> {
    let path = path.as_ref();
    let gltf::Gltf { document, blob } = gltf::Gltf::open(path)?;
    let blob = blob.unwrap_or_else(|| get_buffer(path, &document));

    let meshes = document.meshes().map(|m| get_mesh(m, &blob)).collect();
    let mut textures = vec![];
    let materials = document
        .materials()
        .filter_map(|m| get_material(m, &mut textures, &blob))
        .collect();
    let nodes = document.nodes().filter_map(|n| get_node(n)).collect();

    Ok(Asset {
        meshes,
        materials,
        textures,
        nodes,
    })
}

fn get_node(n: gltf::Node) -> Option<Node> {
    let mesh_id = n.mesh()?.index().into();
    let id = n.index().into();
    let transform = convert_transform(n.transform());
    let children = n.children().map(|c| c.index().into()).collect();

    Some(Node {
        id,
        mesh_id,
        children,
        transform,
    })
}

fn convert_transform(transform: gltf::scene::Transform) -> glam::Affine3A {
    match transform {
        gltf::scene::Transform::Matrix { matrix } => {
            glam::Affine3A::from_mat4(glam::Mat4::from_cols_array_2d(&matrix))
        }
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => glam::Affine3A::from_scale_rotation_translation(
            scale.into(),
            glam::Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
            translation.into(),
        ),
    }
}

fn get_buffer(path: &std::path::Path, document: &gltf::Document) -> Vec<u8> {
    let gltf::buffer::Source::Uri(uri) = document
        .buffers()
        .next()
        .expect("No buffers? Impossible")
        .source()
    else {
        panic!("No buffers? Impossible");
    };

    let mut path = path.to_path_buf();
    path.set_file_name(uri);
    std::fs::read(&path).expect(&format!("Failed to load {path:?} for buffer"))
}

fn get_material(
    material: gltf::Material,
    textures: &mut Vec<Texture>,
    blob: &[u8],
) -> Option<Material> {
    // We don't want the default material
    let Some(id) = material.index() else {
        return None;
    };

    let pbr = material.pbr_metallic_roughness();

    let base_colour_texture = get_texture(
        pbr.base_color_texture().map(|t| t.texture()),
        blob,
        vk::Format::R8G8B8A8_SRGB,
        textures,
    );

    let metallic_roughness_texture = get_texture(
        pbr.metallic_roughness_texture().map(|t| t.texture()),
        blob,
        vk::Format::R8G8B8A8_UNORM,
        textures,
    );

    let normal_texture = get_texture(
        material.normal_texture().map(|t| t.texture()),
        blob,
        vk::Format::R8G8B8A8_UNORM,
        textures,
    );

    let ao_texture = match material.occlusion_texture().map(|t| t.texture().index()) {
        None => NO_TEXTURE.into(),
        // Most of the time, the AO texture is just the red channel of the MR texture
        Some(index) if index == metallic_roughness_texture => metallic_roughness_texture,
        Some(_) => get_texture(
            material.occlusion_texture().map(|t| t.texture()),
            blob,
            vk::Format::R8G8B8A8_UNORM,
            textures,
        ),
    };

    Some(Material {
        id: id.into(),
        base_colour_factor: pbr.base_color_factor().into(),
        emissive_colour_factor: material.emissive_factor().into(),
        base_colour_texture,
        metallic_roughness_texture,
        normal_texture,
        ao_texture,
    })
}

fn load_material(
    material: &Material,
    allocator: &mut Allocator,
    loaded_textures: &HashMap<TextureID, LoadedTexture>,
) -> (MaterialID, LoadedMaterial) {
    let id = material.id;

    // First, we want to patch up all the texture references to point to the loaded texture IDs
    let get = |t| {
        loaded_textures
            .get(t)
            .map(|t| t.id)
            .unwrap_or(NO_TEXTURE.into())
    };

    let material = GPUMaterial {
        base_colour_factor: material.base_colour_factor,
        emissive_colour_factor: material.emissive_colour_factor,
        base_colour_texture: get(&material.base_colour_texture),
        normal_texture: get(&material.normal_texture),
        metallic_roughness_texture: get(&material.metallic_roughness_texture),
        ao_texture: get(&material.ao_texture),
    };

    // Now let's upload it
    let uploaded = allocator.upload_to_slab(&[material]);

    log::trace!(
        "Uploaded material {material:?} to {:?}",
        uploaded.device_address
    );

    (
        id,
        LoadedMaterial {
            id,
            material: uploaded,
        },
    )
}

fn get_texture(
    texture_info: Option<gltf::texture::Texture>,
    blob: &[u8],
    format: vk::Format,
    textures: &mut Vec<Texture>,
) -> TextureID {
    let Some(texture) = texture_info else {
        return NO_TEXTURE.into();
    };

    let gltf::image::Source::View { view, mime_type } = texture.source().source() else {
        panic!("Unsupportd data image type type");
    };

    if mime_type != "image/png" {
        panic!("Unsupported MIME type {mime_type}");
    }

    let image_data = &blob[view.offset()..view.offset() + view.length()];
    let mut decoder = png::Decoder::new(image_data);
    decoder.set_transformations(png::Transformations::ALPHA);
    let mut reader = decoder.read_info().unwrap();

    // Allocate the output buffer.
    let mut buf = vec![0; reader.output_buffer_size()];

    // Read the next frame. An APNG might contain multiple frames.
    let info = reader.next_frame(&mut buf).unwrap();

    // Grab the bytes of the image.
    let data = buf[..info.buffer_size()].to_vec();

    let extent = vk::Extent2D {
        width: info.width,
        height: info.height,
    };
    let id = texture.index().into();

    textures.push(Texture {
        id,
        image_bytes: data,
        extent,
        format,
    });

    id
}

fn load_texture(
    texture: &Texture,
    allocator: &mut Allocator,
    image_manager: &mut ImageManager,
) -> (TextureID, LoadedTexture) {
    let image = image_manager.create_image(
        format!("glTF Texture [#{}]", texture.id.0),
        allocator,
        texture.format,
        texture.extent,
        &texture.image_bytes,
        vk::ImageUsageFlags::SAMPLED,
    );

    (texture.id, image.into())
}

fn get_primitive(primitive: gltf::Primitive, blob: &[u8]) -> Option<Primitive> {
    let reader = primitive.reader(|_| Some(blob));

    // Materials are optional in glTF, but required for us, so grab the ID if it exists or skip
    // this primitive if it doesn't have one.
    let material = primitive.material().index()?.into();

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

    Some(Primitive {
        id: PrimitiveID::next(),
        vertices,
        indices,
        material,
    })
}

fn load_primitive(
    primitive: &Primitive,
    allocator: &mut Allocator,
    vertex_buffer: &mut BufferAllocation<Vertex>,
    index_buffer: &mut BufferAllocation<u32>,
    loaded_materials: &HashMap<MaterialID, LoadedMaterial>,
) -> LoadedPrimitive {
    // Grab the material
    let material = loaded_materials
        .get(&primitive.material)
        .unwrap()
        .material
        .device_address;

    // Upload the indices
    let index_buffer_offset = index_buffer.current_size();
    allocator.append_to_buffer(&primitive.indices, index_buffer);

    log::debug!(
        "[Primitive#{}] Wrote {} indices to [index buffer (device address: {})] at offset {}. First index: {:?}",
        primitive.id,
        primitive.indices.len(),
        index_buffer.device_address,
        index_buffer_offset,
        primitive.indices.first().as_ref().unwrap(),
    );

    // Upload the indices
    let vertex_buffer_offset = vertex_buffer.current_size();
    allocator.append_to_buffer(&primitive.vertices, vertex_buffer);

    log::debug!(
        "[Primitive#{}] Wrote {} vertices to [vertex buffer (device address: {})] at offset {}. First vertex: {:?}",
        primitive.id,
        primitive.vertices.len(),
        vertex_buffer.device_address,
        vertex_buffer_offset,
        primitive.vertices.first().as_ref().unwrap(),
    );

    LoadedPrimitive {
        id: primitive.id,
        index_buffer_offset,
        index_count: primitive.indices.len() as u32,
        vertex_count: primitive.vertices.len() as u32,
        vertex_buffer_offset,
        material,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lazy_vulkan::ash::vk;

    use super::get_asset;
    use crate::load_asset;

    #[test]
    fn test_get_asset() {
        let asset = get_asset("test_assets/cube.glb").unwrap();
        assert_eq!(asset.meshes.len(), 1);
        assert_eq!(asset.meshes[0].primitives.len(), 1);
        assert_eq!(asset.meshes[0].primitives[0].indices.len(), 2916);
        assert_eq!(asset.meshes[0].primitives[0].vertices.len(), 681);
        assert_eq!(asset.materials.len(), 1);
        assert_eq!(asset.textures.len(), 0);
    }

    #[test]
    fn test_get_unit() {
        let asset = get_asset("test_assets/unit.glb").unwrap();
        assert_eq!(asset.meshes.len(), 1);
        assert_eq!(asset.meshes[0].primitives.len(), 1);
        assert_eq!(asset.meshes[0].primitives[0].indices.len(), 2916);
        assert_eq!(asset.meshes[0].primitives[0].vertices.len(), 681);
        assert_eq!(asset.materials.len(), 1);
        assert_eq!(asset.textures.len(), 0);
    }

    #[test]
    fn test_get_asset_gltf() {
        let asset = get_asset("test_assets/cornellBox.gltf").unwrap();
        assert_eq!(asset.meshes.len(), 9);
        assert_eq!(asset.meshes[0].primitives.len(), 1);
        assert_eq!(asset.meshes[0].primitives[0].indices.len(), 36);
        assert_eq!(asset.meshes[0].primitives[0].vertices.len(), 24);
        assert_eq!(asset.materials.len(), 9);
        assert_eq!(asset.textures.len(), 0);
    }

    #[test]
    fn test_load_asset() {
        let core = Arc::new(lazy_vulkan::Core::headless());
        let context = Arc::new(lazy_vulkan::Context::new_headless(&core));
        let mut lazy_vulkan: lazy_vulkan::LazyVulkan<()> = lazy_vulkan::LazyVulkan::headless(
            core,
            context,
            vk::Extent2D::default(),
            vk::Format::R8G8B8A8_UNORM,
        );
        let mut vertex_buffer = lazy_vulkan
            .renderer
            .allocator
            .allocate_buffer(100_000, vk::BufferUsageFlags::STORAGE_BUFFER);
        let mut index_buffer = lazy_vulkan
            .renderer
            .allocator
            .allocate_buffer(100_000, vk::BufferUsageFlags::STORAGE_BUFFER);

        let asset = load_asset(
            "test_assets/cube.glb",
            &mut lazy_vulkan.renderer.allocator,
            &mut lazy_vulkan.renderer.image_manager,
            &mut vertex_buffer,
            &mut index_buffer,
        )
        .unwrap();
        assert_eq!(asset.meshes.len(), 1);
        assert_eq!(asset.meshes[0].primitives.len(), 1);
    }
}
