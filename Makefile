SHELL := /bin/bash

# Usage:
#   make setup
#   make all
#   make dashboard

PYTHON ?= python3
VENV_DIR ?= venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip
MPLCONFIGDIR ?= /tmp/mpl

.PHONY: help windows-help venv install setup activate data models models-econ models-predictive all all-from-raw dashboard clean

help:
	@echo "ASEAN Policy Dashboard Make Targets"
	@echo ""
	@echo "Setup"
	@echo "  make setup              Create venv and install requirements"
	@echo "  make activate           Show activation command"
	@echo ""
	@echo "Pipeline"
	@echo "  make data               Run data pipeline only"
	@echo "  make models             Run all model steps (OLS + Panel OLS + predictive)"
	@echo "  make models-econ        Run OLS + Panel OLS only"
	@echo "  make models-predictive  Run predictive modeling only"
	@echo "  make all                Run full pipeline (data + models)"
	@echo "  make all-from-raw       Rebuild indicators from raw JSON, then run full pipeline"
	@echo ""
	@echo "App"
	@echo "  make dashboard          Launch Streamlit app"
	@echo "  make windows-help       Show PowerShell commands for Windows users"
	@echo ""
	@echo "Maintenance"
	@echo "  make clean              Remove Python cache files"
	@echo ""
	@echo "Options"
	@echo "  Override defaults, e.g.: make setup PYTHON=python3.12"

windows-help:
	@echo "Windows PowerShell Quick Commands"
	@echo ""
	@echo "  .\\scripts\\dev.ps1 setup"
	@echo "  .\\scripts\\dev.ps1 all"
	@echo "  .\\scripts\\dev.ps1 dashboard"

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment in $(VENV_DIR) using $(PYTHON)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
	else \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	fi

install: venv
	@echo "Installing dependencies"
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@$(VENV_PIP) install -r requirements.txt

setup: install
	@echo ""
	@echo "Setup complete."
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

activate:
	@echo "source $(VENV_DIR)/bin/activate"

data: install
	@$(VENV_PYTHON) scripts/run_pipeline.py --stage data

models: install
	@MPLCONFIGDIR=$(MPLCONFIGDIR) $(VENV_PYTHON) scripts/run_pipeline.py --stage models

models-econ: install
	@MPLCONFIGDIR=$(MPLCONFIGDIR) $(VENV_PYTHON) scripts/run_pipeline.py --stage models --skip-predictive

models-predictive: install
	@MPLCONFIGDIR=$(MPLCONFIGDIR) $(VENV_PYTHON) scripts/run_pipeline.py --stage models --skip-ols --skip-fixed-effects

all: install
	@MPLCONFIGDIR=$(MPLCONFIGDIR) $(VENV_PYTHON) scripts/run_pipeline.py --stage all

all-from-raw: install
	@MPLCONFIGDIR=$(MPLCONFIGDIR) $(VENV_PYTHON) scripts/run_pipeline.py --stage all --build-from-raw-json

dashboard: install
	@$(VENV_PYTHON) -m streamlit run app/streamlit_app.py

clean:
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "Removed Python cache files."
