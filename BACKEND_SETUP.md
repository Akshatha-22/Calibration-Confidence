# 🏗️ Backend Setup Guide

## Quick Start

### Option 1: Docker Compose (Recommended) ⭐
```bash
cd e:\projects\Calibration-Confidence
docker-compose up -d
```

Access:
- Frontend: http://localhost:3000  
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Local Development

#### Prerequisites
- Python 3.11+
- PostgreSQL 14+ running locally
- Redis running locally

#### Setup

1. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

2. **Install dependencies**
```bash
pip install -r backend/requirements.txt
```

3. **Setup database**
```bash
# Create database
psql -U postgres -c "CREATE DATABASE calibration_confidence;"

# Set connection string
set DATABASE_URL=postgresql://postgres:password@localhost:5432/calibration_confidence
```

4. **Initialize database**
```bash
python backend/seed_db.py
```

5. **Start backend server**
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

6. **Start frontend**
```bash
cd frontend
pnpm install
pnpm dev  # Runs on http://localhost:3000
```

---

## 📁 Project Structure

### Backend
```
backend/
├── main.py                    # FastAPI application & routes
├── config.py                  # Configuration settings
├── schemas.py                 # Pydantic models
├── websocket_manager.py       # WebSocket handling
├── seed_db.py                 # Database initialization script
├── requirements.txt           # Python dependencies
├── database/
│   ├── models.py              # SQLAlchemy ORM models
│   ├── session.py             # Database connection
│   └── __init__.py
├── tests/
│   ├── test_api.py
│   ├── test_models.py
│   └── conftest.py
└── __init__.py
```

---

## 🗄️ Database Setup

### PostgreSQL Installation

#### Windows
1. Download from https://www.postgresql.org/download/windows/
2. Run installer
3. Remember the password for `postgres` user

#### macOS
```bash
brew install postgresql@14
brew services start postgresql@14
```

#### Linux (Ubuntu)
```bash
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### Create Database
```bash
# Connect as admin
psql -U postgres

# Create database
CREATE DATABASE calibration_confidence;

# Create user (optional)
CREATE USER calibration WITH PASSWORD 'secure-password';
GRANT ALL PRIVILEGES ON DATABASE calibration_confidence TO calibration;

\q  # Exit psql
```

### Redis Installation

#### Windows
- Download from https://github.com/microsoftarchive/redis/releases
- Run installer

#### macOS
```bash
brew install redis
brew services start redis
```

#### Linux (Ubuntu)
```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
```

---

## 🔑 Environment Variables

Create `.env` file:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/calibration_confidence
REDIS_URL=redis://localhost:6379
ENVIRONMENT=development
LOG_LEVEL=info
SECRET_KEY=your-secret-key
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

---

## ✅ Verify Setup

### Check Backend Health
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", "version": "1.0.0"}
```

### Check API Docs
Visit http://localhost:8000/docs in browser

### Access Database
```bash
psql -U postgres -d calibration_confidence
\dt  # List tables
SELECT * FROM models;  # View models
```

---

## 🚀 API Testing

### Get Models
```bash
curl http://localhost:8000/api/models
```

### Create Alert
```bash
curl -X POST http://localhost:8000/api/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mlp",
    "message": "Test alert",
    "severity": "info"
  }'
```

### Get Dashboard
```bash
curl http://localhost:8000/api/dashboard
```

---

## 📊 Database Migrations (Advanced)

Using Alembic for schema changes:

```bash
# Create migration
alembic revision --autogenerate -m "Add new column"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8000
kill -9 <PID>
```

### Database Connection Failed
- Verify PostgreSQL is running
- Check DATABASE_URL is correct
- Ensure database exists

### WebSocket Connection Failed
- Check backend is running
- Verify WS_URL in frontend
- Check for firewall/proxy issues

### Docker Issues
```bash
# View logs
docker-compose logs backend

# Restart services
docker-compose restart

# Clean rebuild
docker-compose down -v
docker-compose up --build
```

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/14/orm/)
- [PostgreSQL Docs](https://www.postgresql.org/docs/)
- [Redis Commands](https://redis.io/commands/)
- [Pydantic](https://docs.pydantic.dev/)

---

## 📝 Next Steps

1. ✅ Start backend server
2. ✅ Initialize database
3. ✅ Connect frontend to backend
4. ✅ Test WebSocket connection
5. ✅ Deploy to production

See `INTEGRATION_GUIDE.md` for complete integration steps.
