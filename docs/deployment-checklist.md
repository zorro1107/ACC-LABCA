# Phase 4 Deployment Checklist

## Pre-Deployment Checklist

### ✅ Prerequisites
- [ ] Azure subscription verified with appropriate permissions
- [ ] Docker installed and verified locally
- [ ] Azure CLI installed (`az --version`)
- [ ] All code tested and committed by Anurag (Person 3)

### ✅ Local Verification
- [ ] `pip install -r requirements.txt` completes successfully
- [ ] `streamlit run app.py` starts without errors
- [ ] `python -m pytest tests/ -v` passes all 38 tests
- [ ] `ml/model.joblib` loads successfully
- [ ] Model prediction returns valid values (0-100%)
- [ ] All three strategies (reactive, predictive, greenscale) produce results
- [ ] Fallback behavior works when model is unavailable

### ✅ Docker Build
- [ ] Dockerfile created at project root
- [ ] `docker build -t greenscale-app:latest .` succeeds
- [ ] `docker run -p 8501:8501 greenscale-app:latest` works locally
- [ ] Dashboard accessible at `http://localhost:8501`
- [ ] CPU history charts render
- [ ] Strategy comparison works
- [ ] Job scheduling decisions display correctly
- [ ] CSV export functions
- [ ] Fallback behavior works when model missing in container

### ✅ Azure Deployment
- [ ] Resource group created in chosen region
- [ ] Container Apps environment created
- [ ] Container image pushed to Azure Container Registry (ACR)
- [ ] Container app deployed with:
  - Target port: 8501
  - Ingress: External
  - CPU: 1 vCPU
  - Memory: 2.0 GiB
  - Min instances: 1
  - Max instances: 3
- [ ] Application accessible via HTTPS URL
- [ ] Dashboard loads correctly in browser
- [ ] All features work identically to local deployment

### ✅ Cloud Evidence Collection
- [ ] Deployed application URL saved
- [ ] Screenshot: Container Apps overview page
- [ ] Screenshot: Container configuration (image, port)
- [ ] Screenshot: Ingress settings (external, port 8501)
- [ ] Screenshot: Scaling configuration (min/max instances)
- [ ] Screenshot: Log stream showing successful startup
- [ ] Screenshot: Metrics page (CPU, memory, HTTP requests)
- [ ] Test results from deployed URL noted

### ✅ Documentation
- [ ] README.md updated with deployment instructions
- [ ] Evidence document created with screenshots
- [ ] Cost estimates documented
- [ ] Scaling limitations documented
- [ ] Disclaimer about simulation vs real cloud data included

### ✅ Cleanup Planning
- [ ] Cost monitoring plan documented
- [ ] Cleanup commands prepared
- [ ] Post-demonstration scaling instructions provided

## Deployment Steps Summary

1. **Build Docker image:**
   ```bash
   docker build -t greenscale-app:latest .
   ```

2. **Create Azure resources:**
   ```bash
   az login
   az group create --name greenscale-rg --location eastus
   az containerapp env create --name greenscale-env --resource-group greenscale-rg --location eastus
   ```

3. **Push image to registry:**
   ```bash
   az acr create --resource-group greenscale-rg --name greenscaleacr --sku Basic
   az acr login --name greenscaleacr
   docker tag greenscale-app:latest greenscaleacr.azurecr.io/greenscale-app:latest
   docker push greenscaleacr.azurecr.io/greenscale-app:latest
   ```

4. **Deploy to Container Apps:**
   ```bash
   az containerapp create \
     --name greenscale-app \
     --resource-group greenscale-rg \
     --environment greenscale-env \
     --image greenscaleacr.azurecr.io/greenscale-app:latest \
     --target-port 8501 \
     --ingress external \
     --cpu 1 --memory 2.0Gi
   ```

5. **Configure scaling:**
   ```bash
   az containerapp update \
     --name greenscale-app \
     --resource-group greenscale-rg \
     --min-instances 1 \
     --max-instances 3
   ```

6. **Get URL:**
   ```bash
   az containerapp show \
     --name greenscale-app \
     --resource-group greenscale-rg \
     --query "properties.configuration.ingress.fqdn" \
     -o tsv
   ```

## Demonstration Script

### Scenario 1: Normal Workload
1. Open deployed URL
2. Select "Normal Demand" preset
3. Run simulation
4. Show stable predictions and balanced strategy comparison

### Scenario 2: Sudden Spike
1. Switch to "Traffic Spike" preset
2. Run simulation
3. Show high CPU prediction
4. Show GreenScale selecting lower-carbon time slot

### Scenario 3: Flexible Job
1. Add flexible job with 30-minute deadline
2. Set up carbon slots with a cleaner future slot
3. Run simulation
4. Show job scheduled in low-carbon slot

### Scenario 4: Urgent Job
1. Add urgent job
2. Run simulation
3. Show job executes immediately at minute 0

### Scenario 5: Invalid Prediction / Fallback
1. Temporarily move or rename model file
2. Restart app or run test
3. Show reactive fallback behavior displayed
4. Restore model file

## Important Notes

- **Native Azure Scaling vs GreenScale:** Azure Container Apps autoscales based on HTTP requests, CPU, or memory. GreenScale's ML recommendations are a simulation and do not control Azure's native scaling behavior.
- **Simulated Data:** All CPU history, carbon intensity values, and energy/emissions estimates in the dashboard are simulated, not actual Azure telemetry.
- **Cost Awareness:** Container Apps bill per-second for compute. Scale to minimum or delete resources after demonstration to avoid charges.