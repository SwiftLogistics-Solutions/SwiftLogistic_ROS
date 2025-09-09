from fastapi import FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware

# Create simple FastAPI application
app = FastAPI(
    title="SwiftTrack Route Optimization System",
    description="Simple API for driver order acceptance and route optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hardcoded test data
WMS_LOCATION = {
    "address": "123 Warehouse Street, Colombo 07",
    "latitude": 6.9271,
    "longitude": 79.8612
}

DRIVERS = {
    "D001": {
        "driver_id": "D001",
        "name": "Sunil Perera",
        "phone": "+94771234567",
        "current_location": {
            "address": "456 Driver Street, Colombo 03",
            "latitude": 6.9147,
            "longitude": 79.8731
        }
    },
    "D002": {
        "driver_id": "D002", 
        "name": "Kamala Gunasekara",
        "phone": "+94771234568",
        "current_location": {
            "address": "789 Driver Avenue, Kandy",
            "latitude": 7.2906,
            "longitude": 80.6337
        }
    }
}

ORDERS = {
    "O001": {
        "order_id": "O001",
        "customer_name": "Nimal Rajapaksa",
        "customer_phone": "+94711234567",
        "delivery_address": {
            "address": "101 Customer Lane, Galle",
            "latitude": 6.0535,
            "longitude": 80.2210
        },
        "priority": "high",
        "status": "pending",
        "accepted_by_driver": None
    },
    "O002": {
        "order_id": "O002",
        "customer_name": "Saman Kumara",  
        "customer_phone": "+94711234568",
        "delivery_address": {
            "address": "202 Customer Road, Negombo",
            "latitude": 7.2083,
            "longitude": 79.8358
        },
        "priority": "low",
        "status": "pending", 
        "accepted_by_driver": None
    },
    "O003": {
        "order_id": "O003",
        "customer_name": "Priyanka Wickramasinghe",
        "customer_phone": "+94711234569", 
        "delivery_address": {
            "address": "303 Customer Plaza, Kandy",
            "latitude": 7.2906,
            "longitude": 80.6337
        },
        "priority": "high",
        "status": "pending",
        "accepted_by_driver": None
    }
}

def calculate_distance(lat1, lng1, lat2, lng2):
    """Simple distance calculation using Haversine formula"""
    import math
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)
    
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def calculate_weighted_distance(physical_distance, priority, priority_weight=0.7):
    """
    Calculate weighted distance considering priority
    High priority orders get lower weights (more attractive routes)
    """
    priority_multipliers = {
        "high": 0.5,  # High priority = 50% weight (shorter weighted distance)
        "low": 1.0    # Low priority = 100% weight (normal weighted distance)
    }
    
    priority_factor = priority_multipliers.get(priority, 1.0)
    distance_factor = 1 - priority_weight
    
    # Weighted distance = (physical_distance * distance_weight) + (physical_distance * priority_factor * priority_weight)
    weighted_distance = (physical_distance * distance_factor) + (physical_distance * priority_factor * priority_weight)
    
    return weighted_distance

def dijkstra_shortest_path(graph, start_node, target_nodes):
    """
    Dijkstra's algorithm to find shortest weighted paths
    Returns: {target_node: (distance, path)}
    """
    import heapq
    
    # Initialize distances and previous nodes
    distances = {node: float('infinity') for node in graph}
    distances[start_node] = 0
    previous = {}
    visited = set()
    
    # Priority queue: (distance, node)
    pq = [(0, start_node)]
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        # Check all neighbors
        for neighbor, weight in graph[current_node].items():
            if neighbor not in visited:
                new_distance = current_distance + weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (new_distance, neighbor))
    
    # Build results for target nodes
    results = {}
    for target in target_nodes:
        if target in distances and distances[target] != float('infinity'):
            # Reconstruct path
            path = []
            current = target
            while current in previous:
                path.append(current)
                current = previous[current]
            path.append(start_node)
            path.reverse()
            
            results[target] = (distances[target], path)
    
    return results

def build_weighted_graph(driver_orders, driver_location, wms_location):
    """
    Build a weighted graph with all locations
    """
    graph = {}
    locations = {}
    
    # Add WMS location
    locations['wms'] = (wms_location["latitude"], wms_location["longitude"])
    
    # Add driver location
    locations['driver'] = (driver_location["latitude"], driver_location["longitude"])
    
    # Add order locations
    for order in driver_orders:
        order_id = order["order_id"]
        locations[order_id] = (
            order["delivery_address"]["latitude"],
            order["delivery_address"]["longitude"]
        )
    
    # Initialize graph nodes
    for node in locations:
        graph[node] = {}
    
    # Add edges between all pairs of locations
    for node1 in locations:
        for node2 in locations:
            if node1 != node2:
                lat1, lng1 = locations[node1]
                lat2, lng2 = locations[node2]
                
                # Calculate physical distance
                physical_dist = calculate_distance(lat1, lng1, lat2, lng2)
                
                # For edges TO order locations, use order priority for weighting
                if node2 in [order["order_id"] for order in driver_orders]:
                    order = next(o for o in driver_orders if o["order_id"] == node2)
                    weighted_dist = calculate_weighted_distance(physical_dist, order["priority"])
                else:
                    # For other edges (WMS, driver), use normal distance
                    weighted_dist = physical_dist
                
                graph[node1][node2] = weighted_dist
    
    return graph, locations

def optimize_route_with_dijkstra(driver_orders, driver_location, wms_location):
    """
    Use Dijkstra's algorithm to find optimal delivery route
    Route: Driver Location → WMS (pickup orders) → Optimized deliveries → Return to WMS
    """
    if not driver_orders:
        return [], 0, []
    
    # Build weighted graph
    graph, locations = build_weighted_graph(driver_orders, driver_location, wms_location)
    
    # Route: Driver → WMS (pickup) → Orders → WMS (return)
    route = ['driver']
    total_weighted_distance = 0
    total_physical_distance = 0
    current_node = 'driver'
    
    # First, go to WMS to pickup orders
    route.append('wms')
    lat1, lng1 = locations['driver']
    lat2, lng2 = locations['wms']
    pickup_distance = calculate_distance(lat1, lng1, lat2, lng2)
    total_physical_distance += pickup_distance
    current_node = 'wms'
    
    # Now optimize delivery sequence using Dijkstra's
    unvisited_orders = [order["order_id"] for order in driver_orders]
    
    # Visit orders using Dijkstra's to find best next order each time
    while unvisited_orders:
        # Use Dijkstra to find shortest paths to all unvisited orders
        paths_result = dijkstra_shortest_path(graph, current_node, unvisited_orders)
        
        # Choose the order with minimum weighted distance
        best_order = None
        min_distance = float('infinity')
        
        for order_id, (distance, path) in paths_result.items():
            if distance < min_distance:
                min_distance = distance
                best_order = order_id
        
        if best_order:
            route.append(best_order)
            total_weighted_distance += min_distance
            
            # Calculate physical distance for this segment
            lat1, lng1 = locations[current_node]
            lat2, lng2 = locations[best_order]
            physical_dist = calculate_distance(lat1, lng1, lat2, lng2)
            total_physical_distance += physical_dist
            
            current_node = best_order
            unvisited_orders.remove(best_order)
    
    # Return to WMS
    if current_node != 'wms':
        route.append('wms')
        weight = graph[current_node]['wms']
        total_weighted_distance += weight
        
        # Calculate physical distance back to WMS
        lat1, lng1 = locations[current_node]
        lat2, lng2 = locations['wms']
        physical_dist = calculate_distance(lat1, lng1, lat2, lng2)
        total_physical_distance += physical_dist
    
    return route, total_physical_distance, locations

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "SwiftTrack Route Optimization System", "status": "running"}

@app.post("/orders/accept/{order_id}/{driver_id}")
async def accept_order(
    order_id: str = Path(..., description="Order ID to accept"),
    driver_id: str = Path(..., description="Driver ID accepting the order")
):
    """Driver accepts a particular order"""
    
    # Check if order exists
    if order_id not in ORDERS:
        return {"error": "Order not found", "order_id": order_id}
    
    # Check if driver exists  
    if driver_id not in DRIVERS:
        return {"error": "Driver not found", "driver_id": driver_id}
    
    # Check if order is already accepted
    if ORDERS[order_id]["accepted_by_driver"] is not None:
        return {
            "error": "Order already accepted", 
            "order_id": order_id,
            "accepted_by": ORDERS[order_id]["accepted_by_driver"]
        }
    
    # Accept the order
    ORDERS[order_id]["accepted_by_driver"] = driver_id
    ORDERS[order_id]["status"] = "accepted"
    
    return {
        "message": "Order accepted successfully",
        "order_id": order_id,
        "driver_id": driver_id,
        "driver_name": DRIVERS[driver_id]["name"],
        "customer_name": ORDERS[order_id]["customer_name"],
        "delivery_address": ORDERS[order_id]["delivery_address"]["address"],
        "priority": ORDERS[order_id]["priority"],
        "status": ORDERS[order_id]["status"]
    }

@app.get("/routes/optimize/{driver_id}")
async def optimize_route(driver_id: str = Path(..., description="Driver ID to optimize route for")):
    """Optimizes the best route for the driver"""
    
    # Check if driver exists
    if driver_id not in DRIVERS:
        return {"error": "Driver not found", "driver_id": driver_id}
    
    # Get driver info
    driver = DRIVERS[driver_id]
    
    # Get orders accepted by this driver
    driver_orders = []
    for order_id, order in ORDERS.items():
        if order["accepted_by_driver"] == driver_id:
            driver_orders.append(order)
    
    if not driver_orders:
        return {
            "message": "No orders assigned to driver",
            "driver_id": driver_id,
            "driver_name": driver["name"],
            "route": []
        }
    
    # Use Dijkstra's algorithm to optimize route
    optimal_route, total_distance, locations = optimize_route_with_dijkstra(
        driver_orders, driver["current_location"], WMS_LOCATION
    )
    
    # Build route points from optimal route
    route_points = []
    
    for i, node in enumerate(optimal_route):
        sequence = i + 1
        coords = locations[node]
        
        if node == 'driver':
            # Starting point - driver's current location
            route_points.append({
                "sequence": sequence,
                "location": f"Start - Driver {driver['name']} Location",
                "address": driver["current_location"]["address"],
                "type": "start",
                "coordinates": {"latitude": coords[0], "longitude": coords[1]}
            })
        
        elif node == 'wms':
            prev_coords = locations[optimal_route[i-1]]
            distance = calculate_distance(prev_coords[0], prev_coords[1], coords[0], coords[1])
            
            # Check if this is pickup (after driver) or return (after deliveries)
            if i == 1:  # Second stop = pickup at warehouse
                route_points.append({
                    "sequence": sequence,
                    "location": "WMS Warehouse - Pickup Orders",
                    "address": WMS_LOCATION["address"],
                    "type": "pickup",
                    "coordinates": {"latitude": coords[0], "longitude": coords[1]},
                    "distance_from_previous": round(distance, 2),
                    "orders_to_pickup": [order["order_id"] for order in driver_orders]
                })
            else:  # Final stop = return to warehouse
                route_points.append({
                    "sequence": sequence,
                    "location": "Return to WMS Warehouse",
                    "address": WMS_LOCATION["address"],
                    "type": "end",
                    "coordinates": {"latitude": coords[0], "longitude": coords[1]},
                    "distance_from_previous": round(distance, 2)
                })
        
        else:  # Order delivery
            order = next(o for o in driver_orders if o["order_id"] == node)
            
            if i == 0:  # This shouldn't happen since driver is always first
                route_points.append({
                    "sequence": sequence,
                    "location": f"Delivery to {order['customer_name']}",
                    "address": order["delivery_address"]["address"],
                    "type": "delivery",
                    "order_id": order["order_id"],
                    "customer_name": order["customer_name"],
                    "customer_phone": order["customer_phone"],
                    "priority": order["priority"],
                    "coordinates": {"latitude": coords[0], "longitude": coords[1]},
                    "priority_weight_applied": "Yes" if order["priority"] == "high" else "No"
                })
            else:
                prev_coords = locations[optimal_route[i-1]]
                distance = calculate_distance(prev_coords[0], prev_coords[1], coords[0], coords[1])
                route_points.append({
                    "sequence": sequence,
                    "location": f"Delivery to {order['customer_name']}",
                    "address": order["delivery_address"]["address"],
                    "type": "delivery",
                    "order_id": order["order_id"],
                    "customer_name": order["customer_name"],
                    "customer_phone": order["customer_phone"],
                    "priority": order["priority"],
                    "coordinates": {"latitude": coords[0], "longitude": coords[1]},
                    "distance_from_previous": round(distance, 2),
                    "priority_weight_applied": "Yes" if order["priority"] == "high" else "No"
                })
    
    # Calculate estimated time (assuming average speed of 40 km/h)
    estimated_time_hours = total_distance / 40
    estimated_time_minutes = int(estimated_time_hours * 60) + (len(driver_orders) * 15)  # Add 15 min per delivery
    
    return {
        "message": "Route optimized using Dijkstra's algorithm",
        "driver_id": driver_id,
        "driver_name": driver["name"],
        "total_orders": len(driver_orders),
        "total_distance_km": round(total_distance, 2),
        "estimated_time_minutes": estimated_time_minutes,
        "route": route_points
    }
