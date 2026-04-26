# 🚀 Backend-Frontend Integration Guide

## Overview

This guide covers the complete integration of your Next.js frontend with the Python FastAPI backend, PostgreSQL database, and real-time WebSocket updates for the Calibration Confidence project.

---

## 📋 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js Frontend (React)                 │
│                  (localhost:3000)                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Dashboard Component (calib            │ │  Zustand Store (dashboardStore.connected.ts)          │ │  └────────────────────────────────────────────────────────┘ │
│  └────────────────────────────────────────────────────────┘ │
│                            │                                   │
│            HTTP API + WebSocket                               │
│                            │                                   │
└──────────────────────────────┼──────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌─────────────────────┐  ┌──────────────┐  ┌──────────────────┐
│   FastAPI Server    │  │  PostgreSQL  │  │  Redis Cache     │
│  (localhost:8000)   │  │ (port:5432)  │  │  (port:6379)     │
│                     │  │              │  │                  │
│  • /api/models      │◄─►  models      │  │  • Pub/Sub       │
│  • /api/alerts      │  │  metrics     │  │  • Session Cache │
│  • /api/dashboard   │  │  alerts      │  │  • Rate Limiting │
│  • /ws/metrics      │  │  predictions │  │                  │
└─────────────────────┘  └──────────────┘  └──────────────────┘
```

---

## 🛠️ Quick Start (Docker Compose)

### Prerequisites
- Docker & Docker Compose
- Node.js 20 (for local development)
- Python 3.11 (for local development)

### Steps

#### 1. Setup Environment
```bash
cd e:\projects\Calibration-Confidence

# Copy environment template
cp .env.example .env

# Update /.env if needed (default values work for local development)
```

#### 2. Start Services
```bash
# Start all services with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
```

#### 3. Initialize Database
```bash
# The database is auto-initialized on backend startup
# Check backend health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "1.0.0"}
```

#### 4. Access Services
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

#### 5. Stop Services
```bash
docker-compose down

# Remove volumes (careful - deletes data)
docker-compose down -v
```

---

## 📦 Project Structure

### Backend (`backend/`)
```
backend/
├── main.py                          # FastAPI application
├── schemas.py                       # Pydantic models for API validation
├── websocket_manager.py             # WebSocket connection management
├── requirements.txt                 # Python dependencies
├── database/
│   ├── models.py                    # SQLAlchemy database models
│   └── session.py                   # Database connection & session
└── tests/
    └── test_api.py                  # API tests
```

### Frontend (`frontend/`)
```
frontend/
├── app/
│   ├── page.tsx                     # Main dashboard page
│   └── layout.tsx                   # App layout
├── components/
│   ├── Dashboard.tsx                # Dashboard component
│   └── ui/                          # shadcn/ui components
├── store/
│   ├── dashboardStore.ts            # Old mock store (backup)
│   └── dashboardStore.connected.ts  # New connected store
├── hooks/
│   └── use-mobile.ts
└── lib/
    └── utils.ts                     # Utility functions
```

---

## 🔌 API Endpoints Reference

### Models

#### List Models
```http
GET /api/models
```
**Response:**
```json
[
  {
    "id": "mlp",
    "name": "Standard MLP",
    "architecture": "mlp",
    "health_score": 95,
    "failure_risk": "low",
    "updated_at": "2026-04-19T10:30:00Z"
  }
]
```

#### Get Model Detail
```http
GET /api/models/{model_id}
```
**Response:**
```json
{
  "id": "mlp",
  "name": "Standard MLP",
  "architecture": "mlp",
  "health_score": 95,
  "failure_risk": "low",
  "ece_25": 0.01,
  "ece_50": 0.03,
  "ece_75": 0.06,
  "calibration_error": 0.03,
  "rmse": 0.11,
  "r_square": 0.92,
  "accuracy": 0.94,
  "grad_ece_correlation": 0.05,
  "ece_flat": true,
  "failure_time_predicted": null,
  "insights": ["Stable performance."],
  "app_context": {
    "name": "Equity Block Trading",
    "exposure": 110.1,
    "metric_name": "Volume (Shares)",
    "metric_value": 154000
  },
  "updated_at": "2026-04-19T10:30:00Z"
}
```

#### Create Model
```http
POST /api/models
Content-Type: application/json

{
  "id": "new-model",
  "name": "New Model",
  "architecture": "lstm",
  "checkpoint_path": "checkpoints/new_model.pth",
  "app_context": {
    "name": "Trading Strategy",
    "exposure": 50.0,
    "metric_name": "Positions Open",
    "metric_value": 125
  }
}
```

#### Get Historical Metrics
```http
GET /api/models/{model_id}/metrics?limit=100
```

### Metrics

#### Update Model Metrics
```http
POST /api/models/{model_id}/metrics
Content-Type: application/json

{
  "health_score": 92,
  "failure_risk": "low",
  "failure_time_predicted": null,
  "ece_25": 0.02,
  "ece_50": 0.05,
  "ece_75": 0.08,
  "calibration_error": 0.04,
  "rmse": 0.12,
  "r_square": 0.89,
  "accuracy": 0.91,
  "grad_ece_correlation": 0.1,
  "ece_flat": true
}
```

### Alerts

#### Get Alerts
```http
GET /api/alerts
```

#### Create Alert
```http
POST /api/alerts
Content-Type: application/json

{
  "model_id": "lstm",
  "message": "Calibration error exceeded threshold",
  "severity": "warning"
}
```

### Dashboard

#### Get Complete Dashboard State
```http
GET /api/dashboard
```
**Response:** All models + alerts for initial page load

### Simulation

#### Trigger Failure Scenario
```http
POST /api/simulate/trigger-failure?model_id=lstm
```

---

## 🔌 WebSocket Connection

### Connection
```typescript
// From dashboardStore.connected.ts
ws = new WebSocket('ws://localhost:8000/ws/metrics');
```

### Message Types

#### 1. Metrics Update
```json
{
  "type": "metrics_update",
  "model_id": "mlp",
  "health_score": 95,
  "failure_risk": "low",
  "timestamp": "2026-04-19T10:30:00Z"
}
```

#### 2. Alert Event
```json
{
  "type": "alert",
  "id": "alert-123",
  "model_id": "lstm",
  "message": "Calibration error increasing",
  "severity": "warning",
  "timestamp": "2026-04-19T10:30:00Z"
}
```

#### 3. Failure Scenario
```json
{
  "type": "failure_scenario",
  "model_id": "lstm",
  "alert_id": "alert-456",
  "health_score": 35,
  "failure_risk": "critical"
}
```

---

## 🔄 Frontend Integration Steps

### Step 1: Replace Store
```bash
# Backup old store
cp frontend/store/dashboardStore.ts frontend/store/dashboardStore.backup.ts

# Use the new connected store
cp frontend/store/dashboardStore.connected.ts frontend/store/dashboardStore.ts
```

### Step 2: Update Component to Connect
Edit `frontend/components/Dashboard.tsx`:

```typescript
import { useEffect } from "react";
import { useDashboardStore } from "@/store/dashboardStore";

export default function Dashboard() {
  const store = useDashboardStore();

  // Load data on mount
  useEffect(() => {
    store.fetchDashboardState();
    
    return () => store.disconnectWebSocket();
  }, [store]);

  // Start simulation timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (store.isSimulating) {
      interval = setInterval(() => {
        store.tickSimulation();
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [store.isSimulating, store.tickSimulation]);

  if (store.isLoading) {
    return <div>Loading...</div>;
  }

  if (store.error) {
    return <div>Error: {store.error}</div>;
  }

  // Rest of your component...
}
```

### Step 3: Environment Configuration
Create `.env.local` in `frontend/`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

---

## 📊 Database Schema

### Models Table
```sql
CREATE TABLE models (
  id VARCHAR PRIMARY KEY,
  name VARCHAR UNIQUE NOT NULL,
  architecture VARCHAR NOT NULL,
  checkpoint_path VARCHAR,
  created_at TIMESTAMP DEFAULT now(),
  updated_at TIMESTAMP DEFAULT now()
);
```

### Model Metrics Table
```sql
CREATE TABLE model_metrics (
  id SERIAL PRIMARY KEY,
  model_id VARCHAR REFERENCES models(id),
  timestamp TIMESTAMP DEFAULT now(),
  health_score FLOAT NOT NULL,
  failure_risk VARCHAR NOT NULL,
  ece_25 FLOAT NOT NULL,
  ece_50 FLOAT NOT NULL,
  ece_75 FLOAT NOT NULL,
  calibration_error FLOAT NOT NULL,
  rmse FLOAT NOT NULL,
  r_square FLOAT NOT NULL,
  accuracy FLOAT NOT NULL,
  grad_ece_correlation FLOAT NOT NULL,
  ece_flat BOOLEAN NOT NULL,
  failure_time_predicted INTEGER,
  calibration_drift FLOAT
);
```

### Alerts Table
```sql
CREATE TABLE alerts (
  id VARCHAR PRIMARY KEY,
  model_id VARCHAR REFERENCES models(id),
  timestamp TIMESTAMP DEFAULT now(),
  message TEXT NOT NULL,
  severity VARCHAR NOT NULL,
  resolved BOOLEAN DEFAULT FALSE
);
```

### App Contexts Table
```sql
CREATE TABLE app_contexts (
  id VARCHAR PRIMARY KEY,
  model_id VARCHAR UNIQUE REFERENCES models(id),
  name VARCHAR NOT NULL,
  exposure FLOAT NOT NULL,
  metric_name VARCHAR NOT NULL,
  metric_value FLOAT NOT NULL,
  updated_at TIMESTAMP DEFAULT now()
);
```

---

## 🚀 Deployment Guide

### Local Development
```bash
# Terminal 1: Backend
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload

# Terminal 2: Frontend
cd frontend
pnpm install
pnpm dev
```

### Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Production Deployment

#### 1. AWS EC2
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Clone repo
git clone <your-repo>
cd Calibration-Confidence

# Setup
cp .env.example .env
# Edit .env with production values

# Deploy
docker-compose -f docker-compose.yml up -d
```

#### 2. Railway
```bash
# Push to GitHub
git push origin main

# Configure Railway:
# 1. Connect GitHub repo
# 2. Create PostgreSQL plugin
# 3. Deploy frontend & backend separately
# 4. Set environment variables
```

#### 3. DigitalOcean
```bash
# Similar to EC2 setup
# Use Docker Droplet for simplicity
```

---

## 🔐 Security Considerations

### Development
- CORS enabled for localhost
- SQLite for testing
- Debug mode enabled

### Production
- [ ] Change SECRET_KEY in .env
- [ ] Update DATABASE_URL to use strong password
- [ ] Enable HTTPS/TLS
- [ ] Restrict CORS to specific domains
- [ ] Use environment variables for secrets
- [ ] Set DEBUG=False
- [ ] Enable PostgreSQL SSL
- [ ] Use environment manager (.env file)
- [ ] Implement authentication (JWT)
- [ ] Add rate limiting

---

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/
```

### Frontend Tests
```bash
cd frontend
pnpm test
```

### API Manual Testing
```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/api/models

# Create alert
curl -X POST http://localhost:8000/api/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mlp",
    "message": "Test alert",
    "severity": "info"
  }'
```

---

## 📈 Data Flow

### 1. Initial Load
```
User opens Frontend
    ↓
Dashboard.tsx useEffect
    ↓
store.fetchDashboardState()
    ↓
GET /api/dashboard
    ↓
Backend: Query all models + latest metrics + alerts
    ↓
Return JSON to frontend
    ↓
Zustand store updates
    ↓
Components re-render with data
```

### 2. Real-time Updates
```
Backend detects metric change
    ↓
POST /api/models/{id}/metrics
    ↓
Backend broadcasts via WebSocket
    ↓
Broadcast to all connected clients
    ↓
Frontend WebSocket receives
    ↓
store.updateMetrics()
    ↓
Zustand updates state
    ↓
Components re-render
```

### 3. Failure Scenario
```
User clicks "TRIGGER FAILURE SCENARIO"
    ↓
POST /api/simulate/trigger-failure
    ↓
Backend creates critical metrics
    ↓
Backend creates alert
    ↓
Broadcast failure_scenario message
    ↓
Frontend updates model health
    ↓
Frontend adds alert to list
```

---

## 🐛 Troubleshooting

### Backend Won't Start
```bash
# Check logs
docker-compose logs backend

# Common issues:
# - Database not ready: wait 5s and retry
# - Port 8000 in use: lsof -i :8000
# - Missing requirements: pip install -r requirements.txt
```

### WebSocket Not Connecting
```bash
# Check WebSocket URL matches WS_URL
echo $NEXT_PUBLIC_WS_URL

# Check backend logs
docker-compose logs backend | grep -i websocket

# Verify backend is running
curl http://localhost:8000/health
```

### Frontend Shows Mock Data
```bash
# Dashboard store hasn't called fetchDashboardState
# Check browser console for errors
# Verify NEXT_PUBLIC_API_URL is correct

# Test API
curl http://localhost:8000/api/dashboard
```

### Database Connection Failed
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# View postgres logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up postgres -d
```

---

## 📚 Additional Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Zustand Store](https://github.com/pmndrs/zustand)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

## 🤝 Contributing

When adding new features:

1. **Backend**: Add endpoint in `backend/main.py` + schema in `backend/schemas.py`
2. **Frontend**: Update `dashboardStore.connected.ts` to call new endpoint
3. **Database**: Add migration if new tables needed
4. **Tests**: Add tests for new endpoints
5. **Docs**: Update this guide

---

## 📝 License

MIT License - See LICENSE file

---

**Last Updated**: April 19, 2026
**Version**: 1.0.0
