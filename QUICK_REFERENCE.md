# 📋 Quick Reference Card

## 🚀 START HERE

### Fastest Way to Run Everything
```bash
cd e:\projects\Calibration-Confidence
docker-compose up -d
```
**Done!** All services running:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- Docs: http://localhost:8000/docs

---

## 📁 Where Everything Is

### Backend API Code
- `backend/main.py` - All API endpoints (15+)
- `backend/schemas.py` - Data models
- `backend/database/models.py` - Database tables
- `backend/websocket_manager.py` - Real-time updates

### Frontend Code
- `frontend/store/dashboardStore.connected.ts` - Connected state management
- `frontend/components/Dashboard.tsx` - Main dashboard
- `frontend/app/page.tsx` - Home page

### Configuration
- `.env` - Environment variables
- `docker-compose.yml` - All services
- `Dockerfile.backend` - Backend container
- `frontend/Dockerfile` - Frontend container

### Documentation
- `INTEGRATION_GUIDE.md` - Complete setup guide
- `BACKEND_SETUP.md` - Backend-only setup
- `IMPLEMENTATION_SUMMARY.md` - What was built

---

## 🔌 API Quick Reference

```bash
# Models
curl http://localhost:8000/api/models
curl http://localhost:8000/api/models/mlp
curl http://localhost:8000/api/models/mlp/metrics

# Alerts
curl http://localhost:8000/api/alerts
curl -X POST http://localhost:8000/api/alerts \
  -H "Content-Type: application/json" \
  -d '{"model_id":"mlp","message":"test","severity":"info"}'

# Dashboard
curl http://localhost:8000/api/dashboard

# Health
curl http://localhost:8000/health

# API Docs
# Open: http://localhost:8000/docs
```

---

## 📊 Database Tables

| Table | Purpose | Key Columns |
|-------|---------|------------|
| `models` | Model definitions | id, name, architecture |
| `model_metrics` | Time-series data | model_id, timestamp, health_score, ece_* |
| `alerts` | Alert events | model_id, message, severity |
| `app_contexts` | Business context | model_id, exposure, metric_name |
| `predictions` | Individual predictions | model_id, prediction, confidence |

---

## 🔄 Data Flow Cheat Sheet

### 1️⃣ User opens frontend
```
Dashboard.tsx loads
  ↓
useEffect calls fetchDashboardState()
  ↓
GET /api/dashboard
  ↓
Zustand store updates
  ↓
Components render with real data
```

### 2️⃣ Real-time metrics update
```
Backend computes new metrics
  ↓
POST /api/models/{id}/metrics
  ↓
WebSocket broadcast to all clients
  ↓
Frontend receives message
  ↓
store.updateMetrics()
  ↓
Auto re-render
```

### 3️⃣ Alert triggered
```
User clicks TRIGGER FAILURE or system detects issue
  ↓
POST /api/alerts
  ↓
Database insert + WebSocket broadcast
  ↓
Frontend adds to alerts list
  ↓
User sees notification
```

---

## 🚦 Status Checks

### Make sure everything is running:
```bash
# Docker services
docker-compose ps

# Backend health
curl http://localhost:8000/health

# Database
psql -U postgres -d calibration_confidence -c "SELECT COUNT(*) FROM models;"

# Frontend loads
curl http://localhost:3000
```

---

## 🛑 Stop/Restart

```bash
# Stop all services
docker-compose down

# Stop but keep data
docker-compose stop

# Restart services
docker-compose restart

# Clean restart (removes data!)
docker-compose down -v
docker-compose up -d
```

---

## 📱 Frontend Integration Status

✅ **Connected features:**
- ✅ Fetch all models on startup
- ✅ Real-time WebSocket updates
- ✅ Auto-reconnect on disconnect
- ✅ Alert notifications
- ✅ Failure scenario triggering
- ✅ Graceful fallback if backend down

**To use connected store:**
```typescript
// In components or hooks
import { useDashboardStore } from '@/store/dashboardStore';

const store = useDashboardStore();
const models = store.models;
const alerts = store.alerts;
```

---

## 🔐 Security Checklist

For production deployment:
- [ ] Update `.env` with strong passwords
- [ ] Change `SECRET_KEY` in `.env`
- [ ] Enable HTTPS/TLS
- [ ] Restrict CORS origins
- [ ] Setup authentication/JWT
- [ ] Enable rate limiting
- [ ] Setup monitoring/logging
- [ ] Regular backups of PostgreSQL

---

## 🐛 Quick Fixes

### Backend won't start?
```bash
docker-compose logs backend
# Check if port 8000 is available
netstat -ano | findstr :8000
```

### WebSocket not connecting?
```bash
# Verify backend is running
curl http://localhost:8000/health

# Check logs
docker-compose logs backend | grep -i websocket
```

### Database connection failed?
```bash
# Restart database
docker-compose restart postgres

# Reinitialize
docker-compose down -v
docker-compose up -d
```

### Frontend shows mock data?
```bash
# Make sure backend is running
curl http://localhost:8000/api/models

# Check browser console for errors
# Look in DevTools → Console tab
```

---

## 📞 Need Help?

1. **Setup issues** → See `BACKEND_SETUP.md`
2. **Integration questions** → See `INTEGRATION_GUIDE.md`
3. **What was built?** → See `IMPLEMENTATION_SUMMARY.md`
4. **API details** → Visit http://localhost:8000/docs

---

## 🎯 Key Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/models` | List all models |
| GET | `/api/models/{id}` | Get model details |
| GET | `/api/dashboard` | Get full dashboard state |
| POST | `/api/models/{id}/metrics` | Update metrics |
| POST | `/api/alerts` | Create alert |
| GET | `/api/alerts` | List alerts |
| WS | `/ws/metrics` | Real-time updates |
| GET | `/health` | Health check |

---

## 🗂️ File Organization

```
Calibration-Confidence/
├── backend/                    # Python FastAPI
│   ├── main.py                # API routes
│   ├── schemas.py             # Data models
│   ├── config.py              # Settings
│   └── database/              # ORM & DB
├── frontend/                  # Next.js React
│   ├── app/                   # Pages
│   ├── components/            # React components
│   └── store/                 # Zustand state
├── docker-compose.yml         # All services
├── .env                       # Configuration
├── INTEGRATION_GUIDE.md       # Setup guide
├── BACKEND_SETUP.md           # Backend guide
└── IMPLEMENTATION_SUMMARY.md  # What's built
```

---

## 💾 Database Connection String

```
postgresql://postgres:postgres@localhost:5432/calibration_confidence
```

If using Docker Compose:
```
postgresql://postgres:postgres@postgres:5432/calibration_confidence
```

---

## 🌐 URLs at a Glance

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Docs (ReDoc) | http://localhost:8000/redoc |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |

---

## 📦 What's Running

When you do `docker-compose up -d`:

```
✓ PostgreSQL 16     → Persistent database
✓ Redis 7          → Caching & pub/sub
✓ FastAPI Server   → Backend API (port 8000)
✓ Next.js App      → Frontend (port 3000)
```

All with health checks and auto-restart on failure.

---

## 🎓 Technologies Used

```
🐍 Backend:     Python 3.11 + FastAPI
🗄️ Database:    PostgreSQL 16
⚡ Cache:       Redis 7
⚛️ Frontend:    Next.js 16 + React 19
📱 State:       Zustand
🚀 Container:   Docker + Docker Compose
🔌 Real-time:   WebSocket
📚 Types:       TypeScript + Pydantic
```

---

## ✅ Verification Checklist

After running `docker-compose up -d`:

- [ ] `docker-compose ps` shows 4 containers running
- [ ] `curl http://localhost:8000/health` returns healthy
- [ ] `curl http://localhost:8000/api/models` returns model list
- [ ] Frontend at http://localhost:3000 loads
- [ ] Dashboard shows 5 models
- [ ] Browser console has no errors

---

**Last Updated**: April 19, 2026  
**Quick Reference v1.0**  
**Status**: ✅ Ready to Deploy
