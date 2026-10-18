# Phase 4: Azure Deployment & Monitoring

## Person 4 - Deployment Engineer

### Overview
This document outlines the deployment process for the GreenScale carbon-aware scheduling application on Azure Container Apps.

## Deployment Steps

### 1. Environment Setup
- [ ] Verify Azure subscription access and permissions
- [ ] Check available region quotas
- [ ] Identify deployment resource group

### 2. Container Build Process
```bash
# Build Docker image locally
docker build -t greenscale-app:latest .

# Local testing
docker run -p 8501:8501 greenscale-app:latest
```

### Azure Portal Alternative Deployment Steps

#### Step 1: Login to Azure Portal
1. Go to https://portal.azure.com
2. Sign in with your Azure credentials

#### Step 2: Check Available Subscriptions
1. Click on your profile icon in the top right (your name/email)
2. Select "View account settings"
3. Go to "Subscriptions" tab
4. Check if there's an active subscription you can use
5. If subscription is Disabled, search for "subscriptions" and click "Subscriptions" to see all listed subscriptions and their states (Enabled/Disabled)

#### Step 3: Create Resource Group (via Portal)
1. In the search bar at the top, type "resource groups" and click on it
2. Click "+ Create"
3. Enter:
   - Subscription: Select your active subscription
   - Resource group: `greenscale-rg`
   - Region: `East US`
4. Click "Review + create" then "Create"

#### Step 4: Create Container Apps Environment (via Portal)
1. Click "Create a resource" (+ icon on the left menu)
2. Search for "Container Apps" and select it, then click "Create"
3. Click "Create" again
4. Under Basics tab:
   - Subscription: Your active subscription
   - Resource group: `greenscale-rg` (just created)
   - Name: `greenscale-env`
   - Region: `East US`
5. Click "Review + Create" then "Create"

#### Step 5: Create the Container App (via Portal)
1. Again, click "Create a resource" → Search "Container Apps" → Click "Create"
2. Under Basics tab:
   - Subscription: Your active subscription
   - Resource group: `greenscale-rg`
   - Name: `greenscale-app`
   - Container Apps environment: `greenscale-env`
   - Click "Next: Container"
3. Under Container tab:
   - Image source: "GitHub" or "Azure Container Registry"
   - For GitHub deployment: Connect to your GitHub repo, select branch, and specify the Dockerfile path
   - For ACR deployment: Login to your container registry and select the pushed image
   - Target port: `8501`
4. Under Ingress tab:
   - Enable ingress: Yes
   - Ingress type: "External"
   - Target port: 8501
5. Click "Review + Create" then "Create"

### 3. Azure Container Registry (ACR) Setup
```bash
# Login to Azure
az login

# Create Container Registry
az acr create --resource-group <rg-name> --name <acr-name> --sku Basic

# Login to ACR
az acr login --name <acr-name>

# Tag and push image
docker tag greenscale-app:latest <acr-name>.azurecr.io/greenscale-app:latest
docker push <acr-name>.azurecr.io/greenscale-app:latest
```

### 4. Azure Container Apps Deployment
```bash
# Create Container App Environment
az containerapp env create --name <env-name> --resource-group <rg-name> --location <region>

# Deploy Container App
az containerapp create \
  --name greenscale-app \
  --resource-group <rg-name> \
  --image <acr-name>.azurecr.io/greenscale-app:latest \
  --environment <env-name> \
  --target-port 8501 \
  --ingress 'external' \
  --cpu 1 \
  --memory 2.0Gi
```

### 5. Post-Deployment Verification
- [ ] Verify deployed URL is accessible
- [ ] Test all simulation scenarios via web UI
- [ ] Confirm model loading from cloud environment
- [ ] Validate three strategies display correctly
- [ ] Test CSV download functionality

### 6. Monitoring Setup
- [ ] Configure log streaming
- [ ] Set up health checks
- [ ] Enable metrics collection
- [ ] Set up alerts for failures

### 7. Scaling Configuration
- [ ] Set minimum instances: `--min-instances 0`
- [ ] Set maximum instances: `--max-instances 2`
- [ ] Document Azure native scaling vs GreenScale recommendations

## Required Evidence for Submission
- [ ] Deployed application URL
- [ ] Screenshots of Azure Container App overview
- [ ] Configuration settings screenshot
- [ ] Available monitoring/metrics screenshots
- [ ] Demonstration video or screenshots of all scenarios

## Cost Management
- [ ] Review pricing at https://azure.microsoft.com/pricing/details/container-apps/
- [ ] Scale to minimum instances after demo period
- [ ] Plan for resource cleanup

## Remaining Tasks
1. ✅ Dockerfile created and tested locally (awaiting Docker environment)
2. ✅ README.md updated with deployment instructions
3. ⏳ Azure CLI installed and authenticated
4. ⏳ Azure Container Apps deployed
5. ⏳ Cloud evidence captured
6. ⏳ Monitoring verified

## Limitations & Assumptions
- Application uses simulated workload/carbon data (as designed by other team members)
- Does not integrate with live Azure monitoring APIs
- Scaling recommendations are simulation outputs, not actual Azure scale operations
- Energy/emissions estimates are based on configurable simulation parameters