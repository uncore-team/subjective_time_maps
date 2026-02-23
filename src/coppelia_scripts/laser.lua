--[[
    FAst Sector-based Laser Scanner (Rig Pool, Pyramid Sensors, Packed Output)

    Description:
        - Uses a pre-allocated pool of rigs (dummy + sector sensors) to handle
          many poses in batches.
        - Each rig has NUM_SECTORS pyramid-type proximity sensors.
        - 'laser_get_observations_from_pose' accepts one or many poses and returns
          all distances as a packed float table for fast Lua <-> Python alignement.

    Pose input format:
        - Single pose: {x, y, theta}
        - Multiple poses: {{x1,y1,th1}, {x2,y2,th2}, ...}

    Packed output layout:
        - flat = [p1_s1, p1_s2, ..., p1_sN, p2_s1, ..., pK_sN]
          where:
            K = number of poses
            N = NUM_SECTORS
]]--

sim = require('sim')

------------------------------------------------------------
-- Robot Profiles
------------------------------------------------------------
local profiles = {
    Turtlebot2 = {
        numSectors      = 4,
        totalSpanDeg    = 240.0,
        minScanDistance = 0.01,
        maxScanDistance = 4.0,
    },
    Burger = {
        numSectors      = 4,
        totalSpanDeg    = 240.0,
        minScanDistance = 0.16,
        maxScanDistance = 8.0,
    },
    -- Burger = {
    --     numSectors      = 8,
    --     totalSpanDeg    = 360.0,
    --     minScanDistance = 0.16,
    --     maxScanDistance = 8.0,
    -- },
}

-- Pool size: number of rigs (dummy + sensors) used in batch processing.
local POOL_SIZE = 2

------------------------------------------------------------
-- Globals
------------------------------------------------------------
local cfg           = nil
local NUM_SECTORS   = nil

-- Pool: sensorPool[i] = { container = handle, sensors = {s1, s2, ..., sN} }
local sensorPool    = {}
local obsCollection = -1

-- Visualization handles
local drawingContainer = nil
local pointContainer   = nil

-- Runtime flags (can be overridden externally)
see_lasers           = "false"
see_fictional_lasers = "false"
see_sectors         = "false"

local selfHandle    = -1
local robotHandle   = -1
local robotAlias    = ""

------------------------------------------------------------
-- Utilities
------------------------------------------------------------

local function clear_drawing()
    if drawingContainer then sim.addDrawingObjectItem(drawingContainer, nil) end
    if pointContainer   then sim.addDrawingObjectItem(pointContainer, nil)   end
end

------------------------------------------------------------
-- Public API
------------------------------------------------------------

-- Real scan from the main rig (sensorPool[1]) at the robot pose.
-- Returns a regular Lua table {d1, d2, ..., dN}.
function laser_get_observations()
    local doDraw  = (see_lasers == "true" or see_lasers == true)
    if doDraw then clear_drawing() end
    local dists   = {}
    local sensors = sensorPool[1].sensors
    local check   = sim.checkProximitySensor
    local maxDist = cfg.maxScanDistance
    local coll    = obsCollection
    local _world  = -1

    local _getMatrix = sim.getObjectMatrix
    local _mulVec    = sim.multiplyVector
    local _drawItem  = sim.addDrawingObjectItem

    for i = 1, NUM_SECTORS do
        local res, dist, pt = check(sensors[i], coll)
        if res > 0 then
            dists[i] = dist
            if doDraw and drawingContainer then
                local m       = _getMatrix(sensors[i], _world)
                local ptWorld = _mulVec(m, pt)
                local originZ = m[12]
                _drawItem(
                    drawingContainer,
                    {m[4], m[8], originZ, ptWorld[1], ptWorld[2], originZ}
                )
                if pointContainer then
                    _drawItem(pointContainer, {ptWorld[1], ptWorld[2], originZ})
                end
            end
        else
            dists[i] = maxDist + 1.0
        end
    end
    return dists
end

-- Virtual scan from one or many poses.
-- Input:
--   posesInput:
--       - {x,y,theta}                (single pose)
--       - {{x1,y1,th1}, ...}         (multiple poses)
--
-- Output:
--   sim.packFloatTable(flat), where flat has size:
--       (#poses * NUM_SECTORS)
--   Layout:
--       [p1_s1, p1_s2, ..., p1_sN, p2_s1, ..., pK_sN]
function laser_get_observations_from_pose(posesInput)
    -- If called with nil, return the real scan packed.
    if POOL_SIZE == 1 then
        POOL_SIZE = 2
    end
    local VPOOL_SIZE = POOL_SIZE - 1
    if not posesInput then
        local d = laser_get_observations()
        return sim.packFloatTable(d)
    end

    -- Detect single vs list
    local isSingle = (type(posesInput[1]) == 'number')
    local count    = isSingle and 1 or #posesInput

    -- Cache API functions for speed
    local _setPose  = sim.setObjectPose
    local _checkProx = sim.checkProximitySensor
    local _pool     = sensorPool
    local _coll     = obsCollection
    local _maxD     = cfg.maxScanDistance
    local _numS     = NUM_SECTORS
    local _world    = -1

    -- Visualization
    local doDraw    = (see_fictional_lasers == "true" or see_fictional_lasers == true)
    if doDraw then clear_drawing() end
    local _drawC     = doDraw and drawingContainer or nil
    local _pointC    = doDraw and pointContainer   or nil
    local _getMatrix = sim.getObjectMatrix
    local _mulVec    = sim.multiplyVector
    local _drawItem  = sim.addDrawingObjectItem

    -- Snapshot real state of the main rig
    --local realPose = sim.getObjectPose(_pool[1].container, _world)
    --local fixedZ   = realPose[3]
    local fixedZ = sim.getObjectPose(_pool[1].container, _world)[3]

    -- Flat result: size = count * NUM_SECTORS
    local flat = {}
    local idx  = 1

    -- Process poses in batches of POOL_SIZE rigs
    for batchStart = 1, count, VPOOL_SIZE do
        local batchEnd  = batchStart + VPOOL_SIZE - 1
        if batchEnd > count then batchEnd = count end
        local loopCount = batchEnd - batchStart + 1

        ------------------------------------------------
        -- 1) Burst move rigs to their poses
        ------------------------------------------------
        for k = 1, loopCount do
            local poseIndex = batchStart + k - 1
            local p         = isSingle and posesInput or posesInput[poseIndex]
            local x, y, th  = p[1], p[2], p[3]

            -- Compute quaternion for Z-rotation
            local half = th * 0.5
            local sz   = math.sin(half)
            local cz   = math.cos(half)

            -- Set rig pose: {x, y, z, qx, qy, qz, qw}
            _setPose(
                _pool[k+1].container,
                _world,
                {x, y, fixedZ, 0, 0, sz, cz}
            )
        end

        ------------------------------------------------
        -- 2) Read sensors for each rig in batch
        ------------------------------------------------
        for k = 1, loopCount do
            local rigSensors = _pool[k+1].sensors

            for s = 1, _numS do
                local res, dist, pt = _checkProx(rigSensors[s], _coll)
                if res > 0 then
                    flat[idx] = dist
                    -- Optional drawing
                    if _drawC then
                        local m       = _getMatrix(rigSensors[s], _world)
                        local ptWorld = _mulVec(m, pt)
                        local zFlat   = m[12]
                        _drawItem(
                            _drawC,
                            {m[4], m[8], zFlat, ptWorld[1], ptWorld[2], zFlat}
                        )
                        if _pointC then
                            _drawItem(_pointC, {ptWorld[1], ptWorld[2], zFlat})
                        end
                    end
                else
                    flat[idx] = _maxD + 1.0
                end
                idx = idx + 1
            end
        end
    end

    -- Restore real pose of the main rig
    --_setPose(_pool[1].container, _world, realPose)

    -- Return packed flat data
    return sim.packFloatTable(flat)
end

------------------------------------------------------------
-- System Call: Init
------------------------------------------------------------

function sysCall_init()
    selfHandle  = sim.getObject('..')
    robotHandle = sim.getObject('::')
    robotAlias  = sim.getObjectAlias(robotHandle, 3)

    cfg = profiles[robotAlias]
    if not cfg then
        sim.addLog(sim.verbosity_scripterrors,
            "[Laser] Unknown robot profile: "..tostring(robotAlias)
        )
        return
    end

    NUM_SECTORS = cfg.numSectors
    local spanRad   = math.rad(cfg.totalSpanDeg)
    local sectorRad = spanRad / NUM_SECTORS

    -- Pyramid geometry at far plane
    local sector_width_far  = 2 * cfg.maxScanDistance * math.tan(sectorRad / 2)
    local sector_height_far = 0.001

    local startAngle = -spanRad / 2 + sectorRad / 2

    -- Proximity sensor params
    local intParams = {0,0,0,0,0,0,0,0}
    local floatParams = {
        0.0, cfg.maxScanDistance, 0.0, 0.0,
        sector_height_far, sector_width_far,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    }

    --------------------------------------------------------
    -- Collection: detect everything except robot & laser trees
    --------------------------------------------------------
    obsCollection = sim.createCollection(0)

    -- Remove robot and laser subtree from detection.
    sim.addItemToCollection(obsCollection, sim.handle_all, -1, 0)         -- add all
    sim.addItemToCollection(obsCollection, sim.handle_tree, robotHandle, 1) -- remove robot tree
    sim.addItemToCollection(obsCollection, sim.handle_tree, selfHandle, 1)  -- remove laser subtree

    --------------------------------------------------------
    -- Pool generation: POOL_SIZE rigs
    --------------------------------------------------------

    -- Check sector visualization options
    local seeSectors = (see_sectors == "true" or see_sectors == true)

    -- Pool creation
    for p = 1, POOL_SIZE do
        local container = sim.createDummy(0.001)
        local rigSensors = {}

        -- Only show sector cones for rig 1 (the real laser rig on the robot)
        local options
        if p == 1 and seeSectors then
            options = 1   -- show sensor cones
        else
            options = 5   -- hide sensor cones
        end

        sim.setObjectAlias(container, "Pool_"..p)

        -- Hide / protect dummy (optional)
        pcall(function()
            local props = sim.objectproperty_collapsed | sim.objectproperty_selectable
            sim.setObjectProperty(container, props)
        end)

        if p == 1 then
            -- First rig follows the real laser base
            sim.setObjectParent(container, selfHandle, true)
            sim.setObjectPosition(container, selfHandle, {0,0,0})
            sim.setObjectOrientation(container, selfHandle, {0,0,0})
        else
            -- Other rigs parked out of the way (world origin Z=-10)
            sim.setObjectParent(container, -1, true)
            sim.setObjectPosition(container, -1, {0,0,-10})
        end

        -- Create NUM_SECTORS pyramid proximity sensors around this container
        for i = 1, NUM_SECTORS do
            local s = sim.createProximitySensor(
                sim.proximitysensor_pyramid_subtype,
                16,
                options,  -- 1:explicit handling; 5: explicit handling + hiding sensor cones
                intParams,
                floatParams
            )

            -- No default special properties: we control detection via collection
            sim.setObjectSpecialProperty(s, 0)

            -- Set parent to container
            sim.setObjectParent(s, container, true)
            sim.setObjectPosition(s, sim.handle_parent, {0,0,0})

            -- Orientation: sector fan in XY plane, aligned with robot +X axis
            local yaw = startAngle + (i-1) * sectorRad

            -- First tilt the sensor so that its local +Z axis points along +X (robot forward)
            -- A rotation of +pi/2 around Y sends +Z -> +X
            local mTilt = sim.buildMatrix({0,0,0}, {0, math.pi/2, 0})

            -- Then apply the per-sector yaw rotation around Z to spread the fan
            local mYaw  = sim.buildMatrix({0,0,0}, {0, 0, yaw})

            -- Combined rotation: first tilt, then yaw
            local mFinal = sim.multiplyMatrices(mYaw, mTilt)
            sim.setObjectMatrix(s, sim.handle_parent, mFinal)

            rigSensors[i] = s

            -- Exclude these sensors from being detected
            sim.addItemToCollection(obsCollection, sim.handle_tree, s, 1)
        end

        sensorPool[p] = { container = container, sensors = rigSensors }
    end
    
    -- To get rig yaw offset and achieve the same orientation in both get lasers functions
    rigYawOffset = 0.0

    do
        local _world = -1
        local pose = sim.getObjectPose(sensorPool[1].container, _world) -- {x,y,z,qx,qy,qz,qw}
        local m = sim.buildMatrixQ({0,0,0}, {pose[4], pose[5], pose[6], pose[7]})
        local e = sim.getEulerAnglesFromMatrix(m)
        rigYawOffset = e[3]  -- yaw
    end

    --------------------------------------------------------
    -- Visualization (only created if enabled)
    --------------------------------------------------------
    drawingContainer = nil
    pointContainer   = nil

    if see_lasers == "true" or see_fictional_lasers == "true" or see_sectors == "true" then
        drawingContainer = sim.addDrawingObject(
            sim.drawing_lines, 2, 0, -1, 5000, {1, 0, 0}
        )
        pointContainer = sim.addDrawingObject(
            sim.drawing_points, 8, 0, -1, 5000, {0, 0, 1}
        )
    end
end

------------------------------------------------------------
-- System Call: Cleanup
------------------------------------------------------------

function sysCall_cleanup()
    for i = 1, #sensorPool do
        if sim.isHandle(sensorPool[i].container) then
            sim.removeObjects({sensorPool[i].container})
        end
    end
    sensorPool = {}

    if drawingContainer then
        sim.removeDrawingObject(drawingContainer)
        drawingContainer = nil
    end
    if pointContainer then
        sim.removeDrawingObject(pointContainer)
        pointContainer = nil
    end
end

function sysCall_sensing()
    -- Nothing here: all scanning is done on-demand via the public API.
end
