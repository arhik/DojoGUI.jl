module DojoGUI

using WGPUCore
using WGPUgfx
using Dojo
using Dojo: input_impulse!, clear_external_force!, update_state!
using Dojo: Shape, Node
using AbstractTrees
using DojoEnvironments
using DataStructures

export WorldNode, buildRobot, print_tree, preparePipeline, prepareObject

mutable struct WorldNode{T<:Renderable} <: Renderable
	parent::Union{Nothing, T}
	object::Union{Nothing, T}
	childObjs::Union{Nothing, Vector{T}}
	body::Union{Nothing, Dojo.Node}
end

WGPUgfx.isTextureDefined(wo::WorldNode{T}) where T<:Renderable = isTextureDefined(wo.object)
WGPUgfx.isTextureDefined(::Type{WorldNode{T}}) where T<:Renderable = isTextureDefined(T)
WGPUgfx.isNormalDefined(wo::WorldNode{T}) where T<:Renderable = isNormalDefined(wo.object)
WGPUgfx.isNormalDefined(::Type{WorldNode{T}}) where T<:Renderable = isNormalDefined(T)


Base.setproperty!(wn::WorldNode{T}, f::Symbol, v) where T<:Renderable = begin
	(f in fieldnames(wn |> typeof)) ?
		setfield!(wn, f, v) :
		setfield!(wn.object, f, v)
end

Base.getproperty(wn::WorldNode{T}, f::Symbol) where T<:Renderable = begin
	(f in fieldnames(wn |> typeof)) ?
		getfield(wn, f) :
		getfield(wn.object, f)
end

mutable struct DojoNode{T<:Dojo.Node} <: Dojo.Node{Float64}
	parent::Union{Nothing, T}
	object::Union{Nothing, T}
	childObjs::Union{Nothing, Vector{T}}
end

AbstractTrees.children(t::WorldNode) = t.childObjs


function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, node::WorldNode)
	summary(io, node)
	print(io, "\n")
	renderObj = (node.object != nothing) ? (node.object.renderObj |> typeof) : nothing
	rType = node.object.rType
	body = node.body
	println(io, " rObj : $(renderObj)")
	println(io, " rType : $(rType)")
	println(io, " body : $(summary(body)), $(body.id), $(body.name)")
end

AbstractTrees.children(t::DojoNode) = t.childObjs

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, node::DojoNode)
	summary(io, node)
	parentId = node.parent == nothing ? "NA" : node.parent.object.id
	objectId = node.object == nothing ? "NA" : node.object.id
	name = node.object == nothing ? "NA" : "$(node.object.name)"
	println(io, " parent : $(parentId)")
	println(io, " object : $(name)")
	println(io, " id : $(objectId)")
end

AbstractTrees.printnode(io::IO, node::WorldNode) = Base.show(io, MIME{Symbol("text/plain")}(), node)
AbstractTrees.printnode(io::IO, node::DojoNode) = Base.show(io, MIME{Symbol("text/plain")}(), node)


function setObject!(tNode::WorldNode, obj::Renderable)
	setfield!(tNode, :object, obj)
end

function setObject!(tNode::DojoNode, obj::Dojo.Node)
	setfield!(tNode, :object, obj)
end

function addChild!(tNode::WorldNode, obj::Union{WorldNode, Renderable})
	if (tNode.childObjs == nothing)
		tNode.childObjs = []
	end
	if obj in tNode.childObjs
		return
	end
	if typeof(obj) <: WorldNode
		push!(tNode.childObjs, obj)
	elseif typeof(obj) <: Renderable
		push!(tNode.childObjs, WorldNode(tNode, obj, Renderable[]))
	end
end

function addChild!(tNode::DojoNode, obj::Union{DojoNode, Dojo.Node})
	if (tNode.childObjs == nothing)
		tNode.childObjs = []
	end
	if obj.id in [node.object.id for node in tNode.childObjs]
		return
	end
	if typeof(obj) == DojoNode
		obj.parent = tNode
		push!(tNode.childObjs, obj)
	elseif typeof(obj) <: Dojo.Node
		push!(tNode.childObjs, DojoNode(tNode, obj, Dojo.Node[]))
	end
end

function removeChild!(tNode::WorldNode, obj::Renderable)
	for (idx, node) in enumerate(tNode.childObjs)
		if obj == node
			popat!(tNode.childObjs, idx)
		end
	end
end

function removeChild!(tNode::DojoNode, obj::Dojo.Node)
	for (idx, node) in enumerate(tNode.childObjs)
		if obj == node
			popat!(tNode.childObjs, idx)
		end
	end
end

function findTNode(tNode::DojoNode, idx::Int64; recursion=1)
	if tNode.parent == nothing && length(tNode.childObjs) == 0
		return tNode
	elseif tNode.object.id == idx
		return tNode
	elseif tNode.parent != nothing && length(tNode.childObjs) == 0
		while tNode.parent != nothing
			tNode = tNode.parent
		end
		return tNode
	else
		for node in tNode.childObjs
			nextNode = findTNode(node, idx; recursion=recursion+1)
			if nextNode ==nothing
				continue
			end
			return nextNode
		end
		# return findTNode(tNode.parent, idx; recursion=recursion)
	end
	# @error "Should not have fall through here"
end

function buildTree(mechanism::Mechanism)
	rootNode = DojoNode(nothing, nothing, Dojo.Node[])
	tNode = rootNode
	for (id, joint) in enumerate(mechanism.joints)
		tNode = findTNode(tNode, joint.parent_id)
		pbody = get_body(mechanism, joint.parent_id)
		cbody = get_body(mechanism, joint.child_id)
		tNode.object == nothing && setObject!(tNode, pbody)
		addChild!(tNode, cbody)
	end
	return rootNode
end

function materialize!(node::DojoNode)
	wn = WorldNode(nothing, nothing, Renderable[], nothing)
	setfield!(wn, :body, getfield(node, :object))
	shape = getfield(node, :object).shape
	wo = nothing
	if typeof(shape) <: Dojo.EmptyShape
		mesh = defaultWGPUMesh(joinpath(pkgdir(WGPUgfx), "assets", "sphere.obj"); scale=0.01f0, color=[1.0, 0.5, 0.5, 1.0])
		wo = WorldObject{WGPUMesh}(mesh, RenderType(VISIBLE | SURFACE | BBOX | AXIS), nothing, nothing, nothing, nothing)
	elseif typeof(shape) <: Dojo.Mesh
		mesh = WGPUgfx.defaultWGPUMesh(shape.path)
		wo = WorldObject{WGPUMesh}(mesh, RenderType(VISIBLE | SURFACE | BBOX | AXIS), nothing, nothing, nothing, nothing)
	end
	
	setObject!(wn, wo)
	
	for cnode in node.childObjs
		cn = materialize!(cnode)
		addChild!(wn, cn)	
		setfield!(cn, :parent, wn)
	end
	
	return wn
end

function buildRobot(mechanism::Mechanism)
	tree = buildTree(mechanism)
	materialize!(tree)
end

function get_joint_dims(mechanism::Mechanism)
	jointNames = mechanism.joints .|> (x) -> getproperty(x, :name)
	dims = jointNames .|> (x) -> input_dimension(get_joint(mechanism, x))
	jointDims = OrderedDict{Symbol, Int}(zip(jointNames, dims))
	return jointDims
end

function get_input_dims(mechanism::Mechanism, x::Symbol)
	return get_joint_dims(mechanism)[x]
end

function get_input_dims(mechanism::Mechanism)
	return get_joint_dims(mechanism) |> values
end

function get_input_idx(mechanism::Mechanism, x::Symbol)
	jointDims = get_joint_dims(mechanism)
	idx = 0
	for (k, v) in jointDims
		idx += v
		if k == x
			return idx
		end
	end
	@error "Could not find the symbol"
end

function WGPUgfx.compileShaders!(gpuDevice, scene::Scene, wn::WorldNode; binding=3)
	WGPUgfx.compileShaders!(gpuDevice, scene, wn.object; binding=binding)
	for node in wn.childObjs
		WGPUgfx.compileShaders!(gpuDevice, scene, node; binding=binding)
	end
end

function WGPUgfx.prepareObject(gpuDevice::WGPUCore.GPUDevice, wn::WorldNode)
	WGPUgfx.prepareObject(gpuDevice, wn.renderObj)
	for node in wn.childObjs
		WGPUgfx.prepareObject(gpuDevice, node.renderObj)
	end
end

function WGPUgfx.preparePipeline(gpuDevice::WGPUCore.GPUDevice, renderer::Renderer, wn::WorldNode; binding=2)
	WGPUgfx.preparePipeline(gpuDevice, renderer, wn.renderObj, binding=binding)
	for node in wn.childObjs
		WGPUgfx.preparePipeline(gpuDevice, renderer, node.renderObj; binding=binding)
	end
end

function WGPUgfx.render(renderPass::WGPUCore.GPURenderPassEncoder, renderPassOptions, wn::WorldNode)
	WGPUgfx.render(renderPass, renderPassOptions, wn.renderObj)
	for node in wn.childObjs
		WGPUgfx.render(renderPass, renderPassOptions, node.renderObj)
	end
end

function WGPUgfx.addObject!(renderer::WGPUgfx.Renderer, obj::WorldNode{T}, camera::WGPUgfx.Camera) where T<:Renderable
	scene = renderer.scene
	setup(renderer, obj.object, camera)
	push!(scene.objects, obj)
	for obj in obj.childObjs
		addObject!(renderer, obj, camera)
	end
end

function WGPUgfx.addObject!(renderer::WGPUgfx.Renderer, obj::WorldNode{T}) where T<:Renderable
	scene = renderer.scene
	for camera in scene.cameraSystem
		setup(renderer, obj.object, camera)
	end
	push!(scene.objects, obj.object)
	for obj in obj.childObjs
		addObject!(renderer, obj)
	end
end

end # module DojoGUI
