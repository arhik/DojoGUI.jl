using Revise
using DojoGUI
using Dojo
using Dojo: input_impulse!, clear_external_force!, update_state!
using DojoEnvironments
using AbstractTrees
using WGPUCore
using WGPUgfx
using GLFW
using GLFW: WindowShouldClose, PollEvents, DestroyWindow
using ControlSystemsBase
using LinearAlgebra
using DataStructures
using Rotations


mechanism = get_mechanism(:cartpole3D)


robot = buildRobot(mechanism)


WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Off)
canvas = WGPUCore.defaultCanvas(WGPUCore.WGPUCanvas)
gpuDevice = WGPUCore.getDefaultDevice()


camera = defaultCamera()
light = defaultLighting()
grid = defaultGrid()
axis = defaultAxis()


scene = Scene(
	gpuDevice,
	canvas,
	camera,
	light,
	[],
	repeat([nothing], 4)...
)


attachEventSystem(scene)


addObject!(scene, grid)
addObject!(scene, axis)
addObject!(scene, robot)


swapMatrix = [1 0 0 0; 0 0 1 0; 0 -1 0 0; 0 0 0 1] .|> Float32


function initTransform!(wn::WorldNode{T}) where T<:Renderable
	object = wn.object
	body = wn.body
	shape = body.shape
	if !(typeof(shape) <: Dojo.EmptyShape)
		state = body.state
		trans = WGPUgfx.translate(state.x1 + Dojo.vector_rotate(shape.position_offset, state.q1))
		scale = WGPUgfx.scaleTransform(shape.scale)
		rot = WGPUgfx.rotateTransform(state.q1*shape.orientation_offset)
		object.uniformData = swapMatrix*(trans∘rot∘scale).linear
	end
	for node in wn.childObjs
		initTransform!(node)
	end
end


function setTransform!(tNode::WorldNode, t)
	tNode.object.uniformData = swapMatrix*t
	if tNode.childObjs == nothing
		return
	end
	# for node in tNode.childObjs
		# setTransform!(node, t)
	# end
end


function stepTransform!(wn::WorldNode{T}) where T<:Renderable
	object = wn.object
	body = wn.body
	shape = body.shape
	if !(typeof(shape) <: Dojo.EmptyShape)
		state = body.state
		trans = WGPUgfx.translate(state.x1)
		scale = WGPUgfx.scaleTransform(shape.scale)
		rot = WGPUgfx.rotateTransform(state.q1*shape.orientation_offset)
		setTransform!(wn, (trans∘rot∘scale).linear)
	end
	for node in wn.childObjs
		stepTransform!(node)
	end
end


function stepController!(mechanism)
	try
		controller!(mechanism, 0)
		for joint in mechanism.joints
			input_impulse!(joint, mechanism)
		end
		status = mehrotra!(mechanism, opts=SolverOptions(verbose=false))
		for body in mechanism.bodies
			clear_external_force!(body) 
		end
		if status == :failed
			@error "Solver Status :failed"
		else
			for body in mechanism.bodies
				update_state!(body, mechanism.timestep)
			end
		end
	catch e
		@info e
	end
end


bodies = mechanism.bodies
origin = mechanism.origin

# ### Controller
stateIdxs = [[4:6..., 10:16...]]
x0 = zeros(16)
u0 = zeros(8)

A, B = get_minimal_gradients!(mechanism, x0, u0)
# Q = [1.0, 1.0, 1.0, 0.2, 0.2, 0.8, 0.002, 0.002, 0.002, 0.002] |> diagm
# Q = [0.00001, 0.00001, 0.00001, 0.099, 0.099, 0.099, 0.0001, 0.0001, 0.00004, 0.00004] |> diagm
Q = [0.00001, 0.00001, 0.00001, 0.099, 0.099, 0.099, 0.0001, 0.0001, 0.00004, 0.00004] |> diagm

x_goal = get_minimal_state(mechanism)[stateIdxs...]

actuators = [:left_wheel, :right_wheel]

R = I(length(actuators))
idxs = [DojoGUI.get_input_idx(mechanism, actuator) for actuator in actuators]


function getNode(wn::WorldNode, name::Symbol)
	body = wn.body
	bodyName = body.name
	if bodyName == name
		return wn
	end
	for node in wn.childObjs
		rNode = getNode(node, name)
		if rNode != nothing
			return rNode
		end
	end
end


function controller!(mechanism, k)
	x = get_minimal_state(mechanism)[stateIdxs...]
	K = lqr(Discrete,A[stateIdxs..., stateIdxs...],B[stateIdxs..., [idxs...]],Q,R)
    u = K * (x - x_goal)
    leftWheel = get_joint(mechanism, :left_wheel)
    rightWheel = get_joint(mechanism, :right_wheel)
    set_input!(leftWheel, [u[1]])
    set_input!(rightWheel, [u[2]])
end


function main()
	camera.eye = [1.0, 0.5, 1.0]
	initialize!(mechanism, :cartpole3D; body_position=[0.0, 0.0, 0.0])
	initTransform!(robot)
	Dojo.initialize_simulation!(mechanism)

	try
		while !WindowShouldClose(canvas.windowRef[])
			status = stepController!(mechanism)
			stepTransform!(robot)
			runApp(scene)
			PollEvents()
		end
	finally
		WGPUCore.destroyWindow(canvas)
	end
end


main()
