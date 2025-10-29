"""Kansas City Metropolitan Area County Boundaries

Simplified GeoJSON polygons for KC metro counties.
In production, these would come from official Census TIGER/Line files.
"""

KC_COUNTY_BOUNDARIES = {
    "Jackson County, MO": {
        "type": "Polygon",
        "coordinates": [[
            [-94.5786, 39.1397],
            [-94.2133, 39.1397],
            [-94.2133, 38.8472],
            [-94.5786, 38.8472],
            [-94.5786, 39.1397]
        ]]
    },
    "Clay County, MO": {
        "type": "Polygon",
        "coordinates": [[
            [-94.5786, 39.4297],
            [-94.2133, 39.4297],
            [-94.2133, 39.1397],
            [-94.5786, 39.1397],
            [-94.5786, 39.4297]
        ]]
    },
    "Platte County, MO": {
        "type": "Polygon", 
        "coordinates": [[
            [-94.9636, 39.5697],
            [-94.5786, 39.5697],
            [-94.5786, 39.4297],
            [-94.9636, 39.4297],
            [-94.9636, 39.5697]
        ]]
    },
    "Johnson County, KS": {
        "type": "Polygon",
        "coordinates": [[
            [-94.9636, 39.0219],
            [-94.6133, 39.0219],
            [-94.6133, 38.7369],
            [-94.9636, 38.7369],
            [-94.9636, 39.0219]
        ]]
    },
    "Wyandotte County, KS": {
        "type": "Polygon",
        "coordinates": [[
            [-94.8636, 39.2078],
            [-94.6133, 39.2078],
            [-94.6133, 39.0219],
            [-94.8636, 39.0219],
            [-94.8636, 39.2078]
        ]]
    },
    "Leavenworth County, KS": {
        "type": "Polygon",
        "coordinates": [[
            [-95.2486, 39.4095],
            [-94.8636, 39.4095],
            [-94.8636, 39.1295],
            [-95.2486, 39.1295],
            [-95.2486, 39.4095]
        ]]
    },
    "Cass County, MO": {
        "type": "Polygon",
        "coordinates": [[
            [-94.5786, 38.8472],
            [-94.2133, 38.8472],
            [-94.2133, 38.5247],
            [-94.5786, 38.5247],
            [-94.5786, 38.8472]
        ]]
    },
    "Miami County, KS": {
        "type": "Polygon",
        "coordinates": [[
            [-94.9636, 38.7369],
            [-94.6133, 38.7369],
            [-94.6133, 38.4519],
            [-94.9636, 38.4519],
            [-94.9636, 38.7369]
        ]]
    },
    "Ray County, MO": {
        "type": "Polygon",
        "coordinates": [[
            [-94.2133, 39.5697],
            [-93.7783, 39.5697],
            [-93.7783, 39.1397],
            [-94.2133, 39.1397],
            [-94.2133, 39.5697]
        ]]
    },
    "Lafayette County, MO": {
        "type": "Polygon",
        "coordinates": [[
            [-93.7783, 39.2797],
            [-93.4333, 39.2797],
            [-93.4333, 38.8472],
            [-93.7783, 38.8472],
            [-93.7783, 39.2797]
        ]]
    },
    "Bates County, MO": {
        "type": "Polygon",
        "coordinates": [[
            [-94.5786, 38.5247],
            [-94.2133, 38.5247],
            [-94.2133, 38.0897],
            [-94.5786, 38.0897],
            [-94.5786, 38.5247]
        ]]
    },
    "Linn County, KS": {
        "type": "Polygon",
        "coordinates": [[
            [-94.9636, 38.4519],
            [-94.6133, 38.4519],
            [-94.6133, 38.0169],
            [-94.9636, 38.0169],
            [-94.9636, 38.4519]
        ]]
    },
    "Clinton County, MO": {
        "type": "Polygon",
        "coordinates": [[
            [-94.5786, 39.7597],
            [-94.2133, 39.7597],
            [-94.2133, 39.5697],
            [-94.5786, 39.5697],
            [-94.5786, 39.7597]
        ]]
    },
    "Caldwell County, MO": {
        "type": "Polygon",
        "coordinates": [[
            [-94.2133, 39.9497],
            [-93.7783, 39.9497],
            [-93.7783, 39.5697],
            [-94.2133, 39.5697],
            [-94.2133, 39.9497]
        ]]
    }
}

# Define county classification for coloring
COUNTY_CLASSIFICATION = {
    "urban": ["Jackson County, MO", "Wyandotte County, KS", "Johnson County, KS", "Clay County, MO", "Platte County, MO"],
    "suburban": ["Cass County, MO", "Leavenworth County, KS", "Miami County, KS"],
    "rural": ["Ray County, MO", "Lafayette County, MO", "Bates County, MO", "Linn County, KS", "Clinton County, MO", "Caldwell County, MO"]
}