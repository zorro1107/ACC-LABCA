# Azure Container Apps Deployment Evidence

This document captures the required evidence for the GreenScale deployment on Azure Container Apps.

## Pre-Deployment Verification

### Environment Prerequisites Checklist
- [ ] Azure subscription with Contributor or Owner role
- [ ] Available region (East US, West US 2, etc.)
- [ ] Sufficient quota for Container Apps (minimum 1 instance required)
- [ ] Container registry access (if using private registry)

### Verification Commands
```bash
# Check Azure CLI installation
az version

# List available subscriptions
az account list --output table

# Check available regions for Container Apps
az provider show --namespace Microsoft.ContainerApps --query "resourceTypes[?resourceType=='environmentWrites']capabilities"
```

## Deployment Evidence Template

### 1. Resource Group
- **Name:** greenscale-rg
- **Location:** eastus
- **Deployment timestamp:** [To be filled after deployment]

### 2. Container Apps Environment
- **Name:** greenscale-env
- **Resource group:** greenscale-rg
- **Deployment date:** [To be filled after deployment]

### 3. Container App Details
- **Name:** greenscale-app
- **Resource group:** greenscale-rg
- **Image source:** [Local Docker image or Azure Container Registry URI]
- **Target port:** 8501
- **Ingress:** External
- **CPU allocation:** 1 vCPU
- **Memory:** 2.0 GiB

### 4. Application URL
- **Full URL:** https://greenscale-app.[env-name].[region].azurecontainerapps.io
- **Verified accessible:** [Yes/No - to be filled after deployment]

## Monitoring Evidence

### Logs and Metrics
1. Log stream URL: [Capture from Azure Portal]
2. Application logs: [Screenshot or evidence of startup]
3. HTTP requests: [Evidence of successful requests]
4. Resource utilization: [CPU%, Memory usage]

### Scaling Configuration
- **Minimum instances:** 1
- **Maximum instances:** 3
- **Scale rules:** HTTP-driven (default)
- **Warm-up time:** [Record from Azure Portal]

## Cost Documentation

### Current Pricing Reference
https://azure.microsoft.com/en-in/pricing/details/container-apps/

**Consumption tier:** Pay-per-second for vCPU and memory
**Expected cost at idle (1 replica minimal):** ~$0.05-0.10 per hour
**Expected cost under load (3 replicas):** ~$0.15-0.30 per hour

### Cost Optimization
- Scale to 0 when not in use
- Reduce max instances to 2 for demo purposes
- Use spot instances if available

## Cleanup Instructions

To avoid unnecessary charges, clean up resources after demonstration:

```bash
# Delete the entire resource group
az group delete --name greenscale-rg --yes --no-wait

# Or scale down to minimum
az containerapp update --name greenscale-app --resource-group greenscale-rg --min-instances 0
```

## Important Disclaimers

### Native Scaling vs GreenScale Recommendations
- Azure Container Apps provides **native autoscaling** based on HTTP requests, CPU usage, and custom metrics
- GreenScale's scheduler is a **simulation** that recommends carbon-aware replica allocation
- The simulation does **not** control Azure's native scaling behavior
- All energy and emissions estimates in the dashboard are **simulated values**, not actual Azure metering data

### Simulation vs Real Cloud Data
- CPU utilization history in the dashboard is **simulated** (Normal Demand, Traffic Spike, Low Demand presets)
- Carbon intensity values are **simulated** time slots, not real-time grid data
- Energy calculations use **assumed** watts-per-core values (50 W by default)
- Emissions estimates use simulated carbon intensity, not real cloud provider emissions data