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


mechanism = get_mechanism(:rhea)


robot = buildRobot(mechanism)


WGPUCore.SetLogLevel(WGPUCore.WGPULogLevel_Off)

scene = Scene()
canvas = scene.canvas
gpuDevice = scene.gpuDevice

renderer = getRenderer(scene)

camera = defaultCamera()
light = defaultLighting()
grid = defaultGrid()
axis = defaultAxis()

setfield!(camera, :id, 1)
scene.cameraSystem = CameraSystem([camera,])


attachEventSystem(renderer)


addObject!(renderer, grid)
addObject!(renderer, axis)
addObject!(renderer, robot)


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
			@info "Solver failed with status :failed"
			return false
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
stateIdxs = [4:6..., 11:16...]
x0 = get_minimal_state(mechanism)
u0 = zeros(12)

A, B = get_minimal_gradients!(mechanism, x0, u0)
# Q = [1.0, 1.0, 1.0, 0.2, 0.2, 0.8, 0.002, 0.002, 0.002, 0.002] |> diagm
# Q = [0.0002, 0.0002, 0.0002, 0.099, 0.099, 0.0004, 0.0004, 0.00004, 0.00004] |> diagm
Q = 0.001.*ones(length(stateIdxs)) |> diagm

x_goal = get_minimal_state(mechanism)[stateIdxs]

actuators = [:left_wheel, :right_wheel, :left_hip, :right_hip, :left_knee, :right_knee]

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
	x = get_minimal_state(mechanism)[stateIdxs]
	K = lqr(Discrete,A[stateIdxs, stateIdxs],B[stateIdxs, [idxs...]],Q,R)
    u = K * (x - x_goal)
    leftWheel = get_joint(mechanism, :left_wheel)
    rightWheel = get_joint(mechanism, :right_wheel)
    leftHip = get_joint(mechanism, :left_hip)
    rightHip = get_joint(mechanism, :right_hip)
    leftKnee = get_joint(mechanism, :left_knee)
    rightKnee = get_joint(mechanism, :right_knee)
    set_input!(leftWheel, [u[1]])
    set_input!(rightWheel, [u[2]])
    # set_input!(leftHip, [u[3]])
    # set_input!(rightHip, [u[4]])
    # set_input!(leftKnee, [u[5]])
    # set_input!(rightKnee, [u[6]])
end

function runApp(renderer)
	init(renderer)
	WGPUgfx.render(renderer)
	deinit(renderer)
end

function main()
	camera.eye = [1.0, 0.5, 1.0]
	initialize!(mechanism, :rhea; body_position=[0.0, 0.0, 0.0])
	initTransform!(robot)
	Dojo.initialize_simulation!(mechanism)

	try
		while !WindowShouldClose(canvas.windowRef[])
			stepController!(mechanism)
			# This section manages to attach camera to a body/object
			floatingBase = getNode(robot, :base_link)
			loc = floatingBase.body.state.x1
			rotMat = Matrix{Float32}(I, (4, 4))
			rotMat[1:3, 1:3] .= RotY(pi/3)
			transformMatrix = floatingBase.object.uniformData*rotMat
			stepTransform!(robot)
			runApp(renderer)
			PollEvents()
		end
	finally
		WGPUCore.destroyWindow(canvas)
	end
end


main()
