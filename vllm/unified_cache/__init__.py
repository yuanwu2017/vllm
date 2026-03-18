# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unified KV + Expert Cache System for MoE models.

This module implements a unified cache framework that manages both KV cache
blocks and MoE expert weights under a single eviction/scoring policy across
GPU HBM and CPU RAM. This enables dynamic memory rebalancing between KV and
expert caches based on workload pressure.
"""
