# SwiftTrack Route Optimization Service (ROS)

A microservice that optimizes delivery routes using Dijkstra's algorithm with priority-weighted graphs. Drivers can accept orders and get optimal routes that prioritize high-priority deliveries while minimizing total distance.

## Quick Start

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```
