# This tries hierarchical control strategy ...

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

cube = WorldObject(
	defaultWGPUMesh("$(pkgdir(WGPUgfx))/assets/cube.obj";scale=0.05f0),
	RenderType(SURFACE | AXIS),
	nothing,
	nothing,
	nothing,
	nothing,
)

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
addObject!(scene, cube)
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
		positionController!(mechanism, 0)
		stabilityController!(mechanism, 0)
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
# stateIdxs = [5, 11, 14, 16] # θ, θ\dot, v, ω̇
stabilityStateIdxs = [5, 11, 14, 16] # θ, θ\dot, v, w are stability state indices
# positionStateIdxs = [1, 2, 3] # direct position state indices hack
positionStateIdxs = [13, 15, 14, 16] # lθ, r0 are position state indices
x0 = zeros(16)
u0 = zeros(8)

A, B = get_minimal_gradients!(mechanism, x0, u0)
# Q = [1.0, 1.0, 1.0, 0.2, 0.2, 0.8, 0.002, 0.002, 0.002, 0.002] |> diagm
# Q = [0.1, 0.01, 0.01, 0.01] |> diagm
# Q = [0.00001, 0.00001, 0.00001, 0.0099, 0.0099, 0.0099, 0.0001, 0.0001, 0.00004, 0.00004] |> diagm
# Q = [0.0001, 0.0001, 0.0001, 0.0099, 0.0099, 0.0099, 0.0001, 0.0001, 0.0001, 0.0001] |> diagm
PQ = 1e-4*ones(length(positionStateIdxs)) |> diagm
SQ = 1e-3*ones(length(stabilityStateIdxs)) |> diagm

# Q = [0.1, 0.1, 1.2, 1.2] |> diagm

x_goal = zeros(size(stabilityStateIdxs))
# x_goal[3] = 0.1

actuators = [:left_wheel, :right_wheel]

PR = I(2) # TODO hardcoded
SR = 0.1*I(length(actuators))

idxs = [DojoGUI.get_input_idx(mechanism, actuator) for actuator in actuators]
SK = lqr(Discrete,A[stabilityStateIdxs, stabilityStateIdxs],B[stabilityStateIdxs, [idxs...]],SQ,SR)
PK = lqr(Discrete,A[positionStateIdxs, positionStateIdxs],B[positionStateIdxs, [idxs...]],PQ,PR)

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

leftWheel = get_joint(mechanism, :left_wheel)
rightWheel = get_joint(mechanism, :right_wheel)


function positionController!(mechanism, k)
	p_goal = [1.2, 1.2, 0, 0]
	x = get_minimal_state(mechanism)[positionStateIdxs]
	u = -PK*(x - p_goal)
	u0[idxs] .= [u[1], u[2]]
	set_input!(mechanism, u0)
end

function stabilityController!(mechanism, k)
	x = get_minimal_state(mechanism)[stabilityStateIdxs]
	# vl = 0.1*x[3]# + rand()
	# vr = 0.1*x[4]# + rand()
	# v = (vr + vl)/2
	# w = (vr - vl)/0.2
	# x[end-1] = v
	# x[end] = w
    u = -SK * (x - x_goal)
    u0[idxs] .= u
    set_input!(mechanism, -u0)
end

using CoordinateTransformations

eyeMatrix = [1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 1] .|> Float32

function main()
	camera.eye = [1.0, 0.5, 1.0]
	initialize!(mechanism, :cartpole3D; body_position=[0.0, 0.0, -0.50])
	initTransform!(robot)
	Dojo.initialize_simulation!(mechanism)
	try
		while !WindowShouldClose(canvas.windowRef[])
			status = stepController!(mechanism)
			# This section manages to attach camera to a body/object
			# floatingBase = getNode(robot, :base_link)
			# loc = floatingBase.body.state.x1
			# rotMat = Matrix{Float32}(I, (4, 4))
			# rotMat[1:3, 1:3] .= RotY(pi/3)
			# rotMat = eyeMatrix*rotMat
			# transformMatrix = floatingBase.object.uniformData*rotMat
			# cube.uniformData = transformMatrix
			# camera.lookat = cube.uniformData[1:3, 3]
			# camera.eye = cube.uniformData[1:3, 4]
			
			stepTransform!(robot)
			runApp(scene)
			PollEvents()
		end
	finally
		WGPUCore.destroyWindow(canvas)
	end
end

main()
