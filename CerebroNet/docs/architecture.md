# Architecture Document

## System Architecture

The CerebroNet platform is designed with a decoupled, microservice-oriented architecture running on local infrastructure without cloud dependencies. The system is orchestrated via Docker Compose and leverages industry-standard MLOps tooling.

```mermaid
graph TD
    User([User / Clinician]) --> UI[Streamlit Frontend]
    UI -- REST API --> API[FastAPI Inference Service]
    
    subgraph "Backend Services"
        API
        MLflow[MLflow Tracking Server]
        Airflow[Apache Airflow ETL]
    end
    
    subgraph "Monitoring & Telemetry"
        Prometheus[Prometheus Metrics]
        Grafana[Grafana Dashboards]
    end
    
    subgraph "Data & Pipeline"
        DVC[DVC Data Versioning]
    end

    API --> MLflow
    API --> Prometheus
    Prometheus --> Grafana
    Airflow --> DVC
```

### Component Details
1. **Frontend**: A Streamlit application rendering an interactive, "container-less" UI.
2. **Backend**: FastAPI providing the REST endpoint for the deep learning model.
3. **MLOps Tracking**: MLflow tracking experiments and model artifacts.
4. **Data Engineering**: Apache Airflow automating ETL and DVC managing data artifacts.
5. **Observability**: Prometheus scraping metrics from FastAPI, and Grafana visualizing them.
