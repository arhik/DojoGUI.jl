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

count = 0

mechanism = get_mechanism(:cartpole3D)

robot1 = buildRobot(mechanism)
# robot2 = buildRobot(mechanism)

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
addObject!(scene, robot1)
# addObject!(scene, robot2)

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
		global count
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
			return status
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
PQ = ones(length(positionStateIdxs)) |> diagm
SQ = ones(length(stabilityStateIdxs)) |> diagm

# Q = [0.1, 0.1, 1.2, 1.2] |> diagm

x_goal = zeros(size(stabilityStateIdxs))
# x_goal[3] = 0.1

actuators = [:left_wheel, :right_wheel]

PR = I(2) # TODO hardcoded
SR = I(length(actuators))

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

p_goal = [1, 0.0, 0.0, 0.0]

gx = zeros(2)
gt = 0.0f0

rotz(t) = [cos(t) -sin(t); sin(t) cos(t)]

function positionController!(mechanism, k)
	global p_goal, gx, gt
	x = get_minimal_state(mechanism)
	xp = x[positionStateIdxs]
	xpos = x[1:2]
	vl = 0.1*xp[3] #+ 0.0001*rand()
	vr = -0.1*xp[4] #+ 0.0001*rand()
	v = (vr + vl)/2
	w = (vr - vl)/0.2
	r = clamp(v/w, -1e23, 1e23)
	xICC = [gx[1] - r*sin(gt), gx[2] + r*cos(gt)]
	gx = rotz(w*mechanism.timestep)*(gx .- xICC) + xICC
	gt += w*mechanism.timestep
	@info "State" gt gx|>adjoint
	xp[1:2] .= gx
	xp[3] = v
	xp[4] = w
	u = -0.1*(xp - p_goal)
	@info u
	dv = u[1]
	dw = u[2]
	# δx = v
	@info dv dw
	vle = (2*dv - 0.5*dw)/2
	vre = (2*dv + 0.5*dw)/2
	x_goal[3: 4] .= 100*[vle, -vre]
end

function setPositionKey(scene)
	WGPUCore.setKeyCallback(scene.canvas, 
	 (_, key, scancode, action, mods) -> begin
	 	name = GLFW.GetKeyName(key, scancode)
 		action == GLFW.PRESS && key == GLFW.KEY_UP && 	(x_goal[3:4] = [100.0, -100.0])
 		action == GLFW.PRESS && key == GLFW.KEY_LEFT && 	(x_goal[3:4] = [-100.0, -100.0])
 		action == GLFW.PRESS && key == GLFW.KEY_RIGHT && 	(x_goal[3:4] = [100.0, 100.0])
 		action == GLFW.PRESS && key == GLFW.KEY_DOWN && 	(x_goal[3:4] = [-100.0, 100.0])
 		action == GLFW.RELEASE && (x_goal[3:4] = [0.0, 0.0])
	 end
	)
end

function stabilityController!(mechanism, k)
	x = get_minimal_state(mechanism)[stabilityStateIdxs]
    u = -SK*(x - x_goal)
    u0[idxs] .= -0.01*sign.(u).*[u[1]^2, u[2]^2]  # .+ [0.003, -0.003]
    set_input!(mechanism, u0)
end

using CoordinateTransformations

eyeMatrix = [1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 1] .|> Float32

function score(mechanism)
	global p_goal, pgoal, gx, gy, gt
	x = get_minimal_state(mechanism)
	xpos = x[1:2]
	xs = x[stabilityStateIdxs]
	vl = 0.1*xs[3] #+ 0.001*rand()
	vr = 0.1*xs[4] #+ 0.001*rand()
	v = (vr + vl)/2
	w = (vr - vl)/0.2
	xs[3] = v
	xs[4] = w
	r = (p_goal .- xs)
	xe = p_goal[1:2] - xpos[1:2]
	sum(xe.*xe)
end

bestScore = 99999999
bodyPosition = [0.0, 0, -0.5]


function plotMain()

end

function main()
	setPositionKey(scene)
	global gx, count, gt, bodyPosition
	gx .= bodyPosition[1:2]
	gt = 0.0f0
	x_goal = zeros(size(stabilityStateIdxs))
	camera.eye = [1.0, 0.5, 1.0]
	initialize!(mechanism, :cartpole3D; body_position=(bodyPosition .+ 0.003*randn(3)))
	initTransform!(robot1)
	# initTransform!(robot2)
	Dojo.initialize_simulation!(mechanism)

	global PR, PQ, SR, SQ, bestScore
	PR = (repeat([rand()], 2))[:] |> diagm
	PQ = repeat((rand(2)), inner=2)[:] |> diagm
	SR = (repeat([rand()], 2)) |> diagm
	SQ = repeat((rand(2)), inner=2)[:] |> diagm

	@info "Parameters" diag(PQ) diag(PR) diag(SQ) diag(SR)

	try
		while !WindowShouldClose(canvas.windowRef[])
			status = stepController!(mechanism)
			count += 1
			
			if count > 100000
				status = :failed
			end

			if status == :failed
				count = 0
				gx .= bodyPosition[1:2]
				gt = 0.0f0
				x_goal = zeros(size(stabilityStateIdxs))
				currentScore = score(mechanism)
				# camera.eye = [1.0, 0.5, 1.0]
				initialize!(mechanism, :cartpole3D; body_position=bodyPosition + 0.003*randn(3))
				initTransform!(robot1)
				# initTransform!(robot2)
				Dojo.initialize_simulation!(mechanism)
				if currentScore < bestScore
					bestScore = currentScore
					PR = (repeat([rand()], 2))[:] |> diagm
					PQ = repeat((rand(2)), inner=2)[:] |> diagm
					SR = (repeat([rand()], 2)) |> diagm
					SQ = repeat((rand(2)), inner=2)[:] |> diagm
					@info "Parameters" diag(PQ) diag(PR) diag(SQ) diag(SR)
					@info "Updating bestScore : $bestScore"
				else
					PR = (repeat([rand()], 2))[:] |> diagm
					PQ = repeat((rand(2)), inner=2)[:] |> diagm
					SR = (repeat([rand()], 2)) |> diagm
					SQ = repeat((rand(2)), inner=2)[:] |> diagm
					@info "Parameters" diag(PQ) diag(PR) diag(SQ) diag(SR)
				end
				@info bestScore
			end
			stepTransform!(robot1)
			runApp(scene)
			PollEvents()
		end
	finally
		WGPUCore.destroyWindow(canvas)
	end
end

main()
