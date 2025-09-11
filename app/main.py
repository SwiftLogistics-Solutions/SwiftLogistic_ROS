from fastapi import FastAPI, Path, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import os
from dotenv import load_dotenv
import math
import heapq
import json
import re

# Load district coordinates from JSON file
def load_district_coordinates():
    """Load district coordinates from JSON file"""
    try:
        with open("district_coordinates.json", "r") as f:
            data = json.load(f)
            return data["districts"]
    except FileNotFoundError:
        print("Warning: district_coordinates.json not found. Location detection will not work.")
        return {}
    except Exception as e:
        print(f"Error loading district coordinates: {e}")
        return {}

# Global variable to store district data
DISTRICT_DATA = load_district_coordinates()

def find_coordinates_by_address(address):
    """
    Find coordinates by matching district names or aliases in the address
    Returns: dict with district info or None if not found
    """
    if not DISTRICT_DATA or not address:
        return None
    
    # Convert address to lowercase for case-insensitive matching
    address_lower = address.lower()
    
    # First try to match exact district names
    for district_name, district_info in DISTRICT_DATA.items():
        if district_name.lower() in address_lower:
            return {
                "district": district_name,
                "latitude": district_info["latitude"],
                "longitude": district_info["longitude"],
                "match_type": "exact_district_name",
                "matched_term": district_name
            }
    
    # Then try to match aliases
    for district_name, district_info in DISTRICT_DATA.items():
        for alias in district_info["aliases"]:
            # Use word boundary matching to avoid partial matches
            pattern = r'\b' + re.escape(alias.lower()) + r'\b'
            if re.search(pattern, address_lower):
                return {
                    "district": district_name,
                    "latitude": district_info["latitude"],
                    "longitude": district_info["longitude"],
                    "match_type": "alias",
                    "matched_term": alias,
                    "matched_alias": alias
                }
    
    return None

def get_all_districts():
    """Get list of all available districts"""
    return list(DISTRICT_DATA.keys()) if DISTRICT_DATA else []

def get_district_info(district_name):
    """Get detailed info for a specific district"""
    return DISTRICT_DATA.get(district_name) if DISTRICT_DATA else None

# Pydantic models
class OrderAcceptRequest(BaseModel):
    order_id: str
    driver_id: str

class OrderDeliveryRequest(BaseModel):
    order_id: str
    driver_id: str
    delivery_proof: str = None

class OrderOnDeliveryRequest(BaseModel):
    order_id: str
    driver_id: str

class DriverSignupRequest(BaseModel):
    firebaseUID: str
    name: str
    email: str
    phone: str
    address: str
    latitude: float = None
    longitude: float = None

# Load environment variables
load_dotenv()

# Create simple FastAPI application
app = FastAPI(
    title="SwiftTrack Route Optimization System",
    description="Simple API for driver order acceptance and route optimization",
    version="1.0.0"
)

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://middleware58_db_user:12345@cluster-1.6ci6iel.mongodb.net/")
DATABASE_NAME = "ROS"

# Initialize MongoDB client
client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]

# Collections
drivers_collection = db.drivers
orders_collection = db.orders

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hardcoded WMS location (as requested)
WMS_LOCATION = {
    "address": "123 Warehouse Street, Colombo 07",
    "latitude": 6.9271,
    "longitude": 79.8612
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

def optimize_route_with_dijkstra(driver_orders, driver_location, wms_location, orders_to_pickup=None):
    """
    Use Dijkstra's algorithm to find optimal delivery route
    Route: Driver Location → [WMS (pickup orders if any)] → Optimized deliveries → [Return to WMS if pickup was done]
    """
    if not driver_orders:
        return [], 0, []
    
    # Build weighted graph
    graph, locations = build_weighted_graph(driver_orders, driver_location, wms_location)
    
    # Route starts with driver
    route = ['driver']
    total_weighted_distance = 0
    total_physical_distance = 0
    current_node = 'driver'
    
    # Only go to WMS if there are orders to pickup
    need_pickup = orders_to_pickup and len(orders_to_pickup) > 0
    
    if need_pickup:
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
    
    # Return to WMS only if we went there for pickup
    if need_pickup and current_node != 'wms':
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

@app.get("/districts")
async def get_districts():
    """Get available districts for location mapping"""
    try:
        districts = get_all_districts()
        district_details = {}
        
        for district in districts:
            district_info = get_district_info(district)
            if district_info:
                district_details[district] = {
                    "latitude": district_info["latitude"],
                    "longitude": district_info["longitude"],
                    "aliases": district_info["aliases"]
                }
        
        return {
            "message": "Available districts for automatic location detection",
            "total_districts": len(districts),
            "districts": district_details,
            "usage": "Include any of these district names or aliases in your address during driver signup for automatic coordinate detection"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching districts: {str(e)}")

@app.post("/drivers/signup")
async def driver_signup(request: DriverSignupRequest):
    """Register a new driver in MongoDB after Firebase authentication"""
    try:
        # Check if driver already exists by firebaseUID
        existing_driver = await drivers_collection.find_one({"firebaseUID": request.firebaseUID})
        if existing_driver:
            raise HTTPException(status_code=400, detail="Driver already exists with this Firebase UID")
        
        # Check if driver already exists by email
        existing_email = await drivers_collection.find_one({"email": request.email})
        if existing_email:
            raise HTTPException(status_code=400, detail="Driver already exists with this email")
        
        # Determine coordinates
        latitude = request.latitude
        longitude = request.longitude
        location_info = None
        
        # If coordinates are not provided, try to detect from address
        if latitude is None or longitude is None:
            location_data = find_coordinates_by_address(request.address)
            if location_data:
                latitude = location_data["latitude"]
                longitude = location_data["longitude"]
                location_info = {
                    "detected_district": location_data["district"],
                    "match_type": location_data["match_type"],
                    "matched_term": location_data["matched_term"],
                    "auto_detected": True
                }
                if "matched_alias" in location_data:
                    location_info["matched_alias"] = location_data["matched_alias"]
            else:
                # Could not detect location from address
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": "Could not determine location from address",
                        "message": "Please provide latitude and longitude coordinates, or include a valid Sri Lankan district/city in your address",
                        "available_districts": get_all_districts()[:10],  # Show first 10 as example
                        "total_districts": len(get_all_districts()),
                        "suggestion": "Include districts like Colombo, Kandy, Galle, etc. in your address for automatic detection"
                    }
                )
        else:
            # Coordinates provided manually
            location_info = {"auto_detected": False, "coordinates_provided": "manual"}
        
        # Generate unique driver ID
        def generate_driver_id():
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            import random
            random_suffix = random.randint(100, 999)
            return f"D{timestamp[-6:]}{random_suffix}"
        
        # Create driver document
        driver_data = {
            "firebaseUID": request.firebaseUID,
            "name": request.name,
            "email": request.email,
            "role": "driver",
            "driver_id": generate_driver_id(),
            "phone": request.phone,
            "current_location": {
                "address": request.address,
                "latitude": latitude,
                "longitude": longitude
            },
            "status": "available",
            "assigned_orders": [],
            "completed_orders": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Insert driver into MongoDB
        result = await drivers_collection.insert_one(driver_data)
        
        # Get the inserted driver
        new_driver = await drivers_collection.find_one({"_id": result.inserted_id})
        
        # Convert ObjectId to string for JSON response
        new_driver["_id"] = str(new_driver["_id"])
        
        response = {
            "message": "Driver registered successfully",
            "driver": new_driver
        }
        
        # Add location detection info if available
        if location_info:
            response["location_detection"] = location_info
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering driver: {str(e)}")

@app.post("/orders/accept")
async def accept_order(request: OrderAcceptRequest):
    """Accept an order and assign it to a driver"""
    order_id = request.order_id
    driver_id = request.driver_id
    
    # Check if order exists
    order = await orders_collection.find_one({"order_id": order_id})
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    
    # Check if driver exists
    driver = await drivers_collection.find_one({"driver_id": driver_id})
    if not driver:
        raise HTTPException(status_code=404, detail=f"Driver {driver_id} not found")
    
    # Check if order is already assigned
    if order["status"] != "pending":
        raise HTTPException(status_code=400, detail=f"Order {order_id} is already {order['status']}")
    
    # Update order status and assign to driver
    await orders_collection.update_one(
        {"order_id": order_id},
        {"$set": {"status": "accepted", "driver_id": driver_id}}
    )
    
    # Add order to driver's assignments
    await drivers_collection.update_one(
        {"driver_id": driver_id},
        {"$addToSet": {"assigned_orders": order_id}}
    )
    
    # Get updated data
    updated_order = await orders_collection.find_one({"order_id": order_id})
    
    return {
        "message": f"Order {order_id} successfully accepted by driver {driver_id}",
        "order": {
            "order_id": updated_order["order_id"],
            "customer_name": updated_order["customer_name"],
            "delivery_address": updated_order["delivery_address"],
            "priority": updated_order["priority"],
            "status": updated_order["status"],
            "driver_id": updated_order["driver_id"]
        },
        "driver": driver["name"]
    }

@app.get("/routes/optimize/{driver_id}")
async def optimize_route(driver_id: str = Path(..., description="Driver ID to optimize route for")):
    """Optimizes the best route for the driver"""
    
    # Check if driver exists
    driver = await drivers_collection.find_one({"driver_id": driver_id})
    if not driver:
        return {"error": "Driver not found", "driver_id": driver_id}
    
    # Get orders assigned to this driver (accepted or on_delivery)
    driver_orders_cursor = orders_collection.find({
        "driver_id": driver_id,
        "status": {"$in": ["accepted", "on_delivery"]}
    })
    all_driver_orders = await driver_orders_cursor.to_list(length=None)
    
    # Separate orders by status
    orders_to_pickup = [order for order in all_driver_orders if order["status"] == "accepted"]
    orders_on_delivery = [order for order in all_driver_orders if order["status"] == "on_delivery"]
    
    if not all_driver_orders:
        return {
            "message": "No orders assigned to driver",
            "driver_id": driver_id,
            "driver_name": driver["name"],
            "route": []
        }
    
    # Use only orders that need delivery for route optimization
    driver_orders = all_driver_orders
    
    # Use Dijkstra's algorithm to optimize route
    optimal_route, total_distance, locations = optimize_route_with_dijkstra(
        driver_orders, driver["current_location"], WMS_LOCATION, orders_to_pickup
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
                    "orders_to_pickup": [order["order_id"] for order in orders_to_pickup]
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
                    "status": order["status"],
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
                    "status": order["status"],
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
        "orders_to_pickup": len(orders_to_pickup),
        "orders_on_delivery": len(orders_on_delivery),
        "warehouse_visit_required": len(orders_to_pickup) > 0,
        "total_distance_km": round(total_distance, 2),
        "estimated_time_minutes": estimated_time_minutes,
        "route": route_points
    }

@app.post("/orders/on-delivery")
async def mark_order_on_delivery(request: OrderOnDeliveryRequest):
    """Mark an order as on delivery (picked up from warehouse, en route to customer)"""
    order_id = request.order_id
    driver_id = request.driver_id
    
    # Check if order exists
    order = await orders_collection.find_one({"order_id": order_id})
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    
    # Check if driver exists
    driver = await drivers_collection.find_one({"driver_id": driver_id})
    if not driver:
        raise HTTPException(status_code=404, detail=f"Driver {driver_id} not found")
    
    # Check if order is assigned to this driver
    if order.get("driver_id") != driver_id:
        raise HTTPException(status_code=400, detail=f"Order {order_id} is not assigned to driver {driver_id}")
    
    # Check if order is in accepted status
    if order["status"] != "accepted":
        raise HTTPException(status_code=400, detail=f"Order {order_id} is not in accepted status. Current status: {order['status']}")
    
    # Update order status to on_delivery
    await orders_collection.update_one(
        {"order_id": order_id},
        {"$set": {
            "status": "on_delivery",
            "picked_up_at": datetime.utcnow(),
            "picked_up_by": driver_id
        }}
    )
    
    # Get updated order data
    updated_order = await orders_collection.find_one({"order_id": order_id})
    
    return {
        "message": f"Order {order_id} successfully marked as on delivery by driver {driver_id}",
        "order": {
            "order_id": updated_order["order_id"],
            "customer": updated_order["customer_name"],
            "delivery_address": updated_order["delivery_address"],
            "priority": updated_order["priority"],
            "status": updated_order["status"],
            "driver_id": updated_order["driver_id"],
            "picked_up_at": updated_order["picked_up_at"].isoformat()
        },
        "driver": driver["name"]
    }

@app.post("/orders/deliver")
async def mark_order_delivered(request: OrderDeliveryRequest):
    """Mark an order as delivered"""
    order_id = request.order_id
    driver_id = request.driver_id
    delivery_proof = request.delivery_proof
    
    # Check if order exists
    order = await orders_collection.find_one({"order_id": order_id})
    if not order:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    
    # Check if driver exists
    driver = await drivers_collection.find_one({"driver_id": driver_id})
    if not driver:
        raise HTTPException(status_code=404, detail=f"Driver {driver_id} not found")
    
    # Check if order is assigned to this driver
    if order.get("driver_id") != driver_id:
        raise HTTPException(status_code=400, detail=f"Order {order_id} is not assigned to driver {driver_id}")

    # Check if order is in on_delivery status
    if order["status"] != "on_delivery":
        raise HTTPException(status_code=400, detail=f"Order {order_id} is not in on_delivery status. Current status: {order['status']}")
    
    # Update order status to delivered
    update_data = {
        "status": "delivered",
        "delivered_at": datetime.utcnow(),
        "delivered_by": driver_id
    }
    
    if delivery_proof:
        update_data["delivery_proof"] = delivery_proof
    
    await orders_collection.update_one(
        {"order_id": order_id},
        {"$set": update_data}
    )
    
    # Remove order from driver's assigned orders and add to completed
    await drivers_collection.update_one(
        {"driver_id": driver_id},
        {
            "$pull": {"assigned_orders": order_id},
            "$addToSet": {"completed_orders": order_id}
        }
    )
    
    # Get updated order data
    updated_order = await orders_collection.find_one({"order_id": order_id})
    
    return {
        "message": f"Order {order_id} successfully marked as delivered by driver {driver_id}",
        "order": {
            "order_id": updated_order["order_id"],
            "customer": updated_order["customer_name"],
            "delivery_address": updated_order["delivery_address"],
            "priority": updated_order["priority"],
            "status": updated_order["status"],
            "driver_id": updated_order["driver_id"],
            "delivered_at": updated_order["delivered_at"].isoformat(),
            "delivery_proof": updated_order.get("delivery_proof", "")
        },
        "driver": driver["name"]
    }
