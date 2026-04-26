# 🚀 Deployment Checklist

## Pre-Flight Checks

### Local Testing ✅
- [x] Backend API runs locally
- [x] Frontend connects to backend
- [x] WebSocket updates work
- [x] Database queries work
- [x] All 5 models load correctly

### Code Quality
- [ ] Run backend tests: `pytest backend/tests/`
- [ ] Run frontend linting: `pnpm lint`
- [ ] Check TypeScript: `pnpm type-check`

### Documentation
- [x] INTEGRATION_GUIDE.md complete
- [x] BACKEND_SETUP.md complete
- [x] IMPLEMENTATION_SUMMARY.md complete
- [x] QUICK_REFERENCE.md complete
- [x] API documentation auto-generated at `/docs`

---

## Environment Setup

### Production Environment File (.env)
```bash
# Database (use managed service!)
DATABASE_URL=postgresql://user:securepass@db.example.com:5432/calibration

# Redis (use managed service!)
REDIS_URL=redis://:securepass@cache.example.com:6379

# Environment
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=error

# API
API_HOST=0.0.0.0
API_PORT=8000

# Frontend URLs
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com

# Security
SECRET_KEY=<generate-with-secrets.token_urlsafe(32)>
ALLOWED_ORIGINS=https://app.yourdomain.com,https://yourdomain.com
```

---

## Deployment Options

### Option 1: AWS (EC2 + RDS + ElastiCache)

**1. Setup Compute**
```bash
# Launch Ubuntu 22.04 LTS
# t3.medium instance
# 20GB storage
# Security group: Allow 80, 443, 3000, 8000 from specific IPs
```

**2. Setup Database**
```bash
# RDS PostgreSQL 15
# db.t3.micro (upgrade as needed)
# Multi-AZ for high availability
# Automated backups
# SSL enabled
```

**3. Setup Cache**
```bash
# ElastiCache Redis 7
# cache.t3.micro
# Multi-AZ replication
```

**4. Deploy Application**
```bash
# SSH into EC2
ssh -i key.pem ubuntu@your-ip

# Clone and setup
git clone <repo>
cd Calibration-Confidence

# Update .env with RDS & ElastiCache URLs
nano .env

# Deploy
docker-compose up -d
```

**5. Add Domain & SSL**
```bash
# Use Route53 for DNS
# Use ACM for SSL certificates
# Setup CloudFront CDN (optional)
```

### Option 2: Railway (Recommended for Simplicity)

**1. Connect GitHub**
- Push code to GitHub
- Connect Railway to repo

**2. Create Plugins**
- PostgreSQL plugin
- Redis plugin (optional)

**3. Deploy Services**
- Backend service (Python)
- Frontend service (Node.js)

**4. Configure Environment**
- Railway auto-generates DB URLs
- Set NEXT_PUBLIC_API_URL
- Set SECRET_KEY

**5. Go Live**
- Deploy on push to main

### Option 3: DigitalOcean (App Platform)

Similar to Railway:
1. Connect GitHub repo
2. Create App Platform app
3. Add PostgreSQL database
4. Deploy services
5. Configure domain

---

## Infrastructure Requirements

### Minimum (MVP)
- 2 vCPU
- 2GB RAM
- 20GB SSD
- PostgreSQL 14+
- Redis 6+

### Recommended (Production)
- 4+ vCPU
- 4GB+ RAM
- 50GB+ SSD
- PostgreSQL 15+ (managed)
- Redis 7+ (managed)
- Load balancer
- CDN
- Backups

---

## SSL/TLS Setup

### Using Let's Encrypt with Nginx
```bash
# Install Nginx & Certbot
apt-get install nginx certbot python3-certbot-nginx

# Get certificate
certbot certonly --nginx -d api.yourdomain.com -d yourdomain.com

# Configure Nginx
# (See reverse proxy config below)
```

### Nginx Reverse Proxy Config
```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # WebSocket
    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## Monitoring & Logging

### Application Monitoring
```python
# Add to backend/main.py
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### Logging Setup
```bash
# View logs
docker-compose logs -f backend

# Send to cloud
# - CloudWatch (AWS)
# - Stackdriver (GCP)
# - Datadog
# - New Relic
```

### Database Monitoring
- Setup alerts for replication lag
- Monitor query performance
- Setup automated backups
- Test restore procedures

---

## Performance Optimization

### Database
```sql
-- Create indexes for common queries
CREATE INDEX idx_models_updated_at ON models(updated_at DESC);
CREATE INDEX idx_metrics_model_timestamp ON model_metrics(model_id, timestamp DESC);
CREATE INDEX idx_alerts_model_severity ON alerts(model_id, severity);
```

### Caching
```bash
# Cache API responses
- GET /api/models: Cache 5 minutes
- GET /api/dashboard: Cache 1 minute
- GET /api/alerts: Cache 30 seconds
```

### Frontend
```bash
# Enable Next.js caching headers
# Configure ISR (Incremental Static Regeneration)
# Enable image optimization
# Setup CDN distribution
```

---

## Security Hardening

### Database
- [ ] Use managed PostgreSQL service
- [ ] Enable SSL/TLS connections
- [ ] Setup VPC security groups
- [ ] Regular backups to S3
- [ ] Enable encryption at rest
- [ ] Setup read replicas

### Application
- [ ] Rotate SECRET_KEY
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS everywhere
- [ ] Setup rate limiting
- [ ] Enable CORS only for known domains
- [ ] Use security headers (HSTS, CSP)
- [ ] Setup WAF (Web Application Firewall)
- [ ] Regular security updates

### Infrastructure
- [ ] SSH key-only access
- [ ] Disable root login
- [ ] Setup fail2ban for brute force
- [ ] Regular security audits
- [ ] Setup intrusion detection

---

## Backup & Disaster Recovery

### Database Backups
```bash
# Automated daily backups (AWS RDS)
# Retention: 30 days
# Cross-region replication

# Manual backup
pg_dump <db_url> > backup.sql
```

### Code Backups
```bash
# Use GitHub with branch protection
# Require code review before merge
# Tag releases
```

### Recovery Procedures
- [ ] Document restore procedures
- [ ] Test restore quarterly
- [ ] Setup alerts for backup failures
- [ ] Maintain runbooks

---

## Performance Benchmarks

### Target Metrics
- API response time: <200ms (p95)
- WebSocket latency: <100ms
- Dashboard load time: <2s
- Database query time: <100ms

### Monitoring Commands
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null \
  -s http://localhost:8000/api/models

# Load test
ab -n 1000 -c 10 http://localhost:8000/api/models
```

---

## CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r backend/requirements.txt
          pytest backend/tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # SSH to server
          # git pull
          # docker-compose up -d
```

---

## Post-Deployment

### Smoke Tests
```bash
# Health check
curl https://api.yourdomain.com/health

# API test
curl https://api.yourdomain.com/api/models

# WebSocket test
# Connect in browser console
# ws = new WebSocket('wss://api.yourdomain.com/ws/metrics');
```

### Monitoring Setup
- [ ] Setup error tracking (Sentry)
- [ ] Setup APM (Application Performance Monitoring)
- [ ] Setup uptime monitoring
- [ ] Setup alerts for errors/downtime
- [ ] Dashboard for key metrics

### Documentation
- [ ] Update README with production URLs
- [ ] Document deployment process
- [ ] Document troubleshooting procedures
- [ ] Maintain operations runbook

---

## Scaling Considerations

### Horizontal Scaling
```bash
# Multiple backend instances with load balancer
# Database connection pooling (PgBouncer)
# Redis cluster for high availability
# CDN for frontend static assets
```

### Vertical Scaling
```bash
# Upgrade instance size as needed
# Increase database instance class
# Add more memory
# Use faster storage (SSD)
```

### Database Optimization
```sql
-- Add read replicas
-- Setup partitioning for large tables
-- Archive old data
-- Optimize indexes
```

---

## Cost Optimization

### AWS Example
| Service | Size | Cost/month |
|---------|------|-----------|
| EC2 | t3.small | $15 |
| RDS | db.t3.micro | $20 |
| ElastiCache | cache.t3.micro | $15 |
| Data transfer | ~10GB | $5 |
| **Total** | | **~$55** |

### Cost Reduction Tips
1. Use spot instances for non-critical workloads
2. Enable auto-scaling
3. Archive old data
4. Use managed services (RDS, ElastiCache)
5. Monitor and kill unused resources

---

## Support & Maintenance

### Daily
- [ ] Monitor error logs
- [ ] Check uptime
- [ ] Verify backups

### Weekly
- [ ] Review performance metrics
- [ ] Check security alerts
- [ ] Review new issues

### Monthly
- [ ] Security audit
- [ ] Performance review
- [ ] Backup restoration test
- [ ] Dependency updates

### Quarterly
- [ ] Load testing
- [ ] Capacity planning
- [ ] Security audit
- [ ] Disaster recovery drill

---

## Rollback Plan

If deployment fails:
```bash
# Revert to previous image
docker-compose down
git checkout previous-commit
docker-compose up -d

# Or use blue-green deployment
# Keep previous version running
# Switch traffic if needed
```

---

## Handoff Documentation

Before going live:
1. ✅ Architecture diagram
2. ✅ Deployment procedures
3. ✅ Troubleshooting guide
4. ✅ Emergency contacts
5. ✅ Escalation procedures
6. ✅ Performance baselines
7. ✅ Monitoring dashboards
8. ✅ Backup procedures

---

**Deployment Checklist v1.0**  
**Last Updated**: April 19, 2026
