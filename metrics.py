#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines the standard air quality metrics used to validate the POD-GPR
surrogate model.

See Chang et Hanna. 2004. Air quality model performance evaluation.

@author: lumet
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np

# =============================================================================
# Auxiliary functions
# =============================================================================
def compute_area_over_iso(y, iso, weights):
    hit = y >= iso
    if weights is None:
        #We assume that all the cells have a unit area area
        area = np.sum(hit)
    else:
        area = np.sum(weights*hit)
    
    return area

def compute_inter_area_over_iso(y_pred, y_ref, iso, weights):
    hit = y_pred >= iso
    hit *= y_ref >= iso  # * corresponds to element wise And
    if weights is None:
        #We assume that all the cells have a unit area
        area = np.sum(hit)
    else:
        area = np.sum(weights*hit)
    
    return area

def compute_union_area_over_iso(y_pred, y_ref, iso, weights):
    hit = y_pred >= iso
    hit += y_ref >= iso  # + corresponds to element wise Or
    if weights is None:
        #We assume that all the cells have a unit area
        area = np.sum(hit)
    else:
        area = np.sum(weights*hit)

    return area

def compute_xor_area_over_iso(y_pred, y_ref, iso, weights):
    union_area = compute_union_area_over_iso(y_pred, y_ref, iso, weights)
    inter_area = compute_inter_area_over_iso(y_pred, y_ref, iso, weights)
    
    return union_area - inter_area

# =============================================================================
# Metrics
# =============================================================================
def fb(y_pred, y_ref, weights=None):
    """
    Fractional Bias.  
    """
    y_pred_mean = np.average(y_pred, weights=weights)
    y_ref_mean = np.average(y_ref, weights=weights)
    
    return (y_ref_mean - y_pred_mean)/(0.5*(y_ref_mean + y_pred_mean))

def nmse(y_pred, y_ref, weights=None):
    """
    Normalized Mean Square Error.
    """
    return np.average((y_ref - y_pred) ** 2, weights=weights)/(
           np.average(y_ref, weights=weights) * np.average(y_pred, weights=weights))

def mg(y_pred, y_ref, tol=1e-16, weights=None):
    """
    Mean Geometric Bias.
    """
    y_pred_trunc = np.where(y_pred<=tol, tol, y_pred)
    y_ref_trunc = np.where(y_ref<=tol, tol, y_ref)
    return np.exp(np.average(np.log(y_ref_trunc), weights=weights) - \
                  np.average(np.log(y_pred_trunc), weights=weights))

def vg(y_pred, y_ref, tol=1e-16, weights=None):
    """
    Geometric Variance.
    
    Chang et Hanna. 2004. Air quality model performance evaluation. 
    """
    y_pred_trunc = np.where(y_pred<=tol, tol, y_pred)
    y_ref_trunc = np.where(y_ref<=tol, tol, y_ref)
    return np.exp(np.average((np.log(y_ref_trunc) - np.log(y_pred_trunc))**2, 
                             weights=weights))

def fac2(y_pred, y_ref, tol=1e-16, weights=None):
    """
    Percentage of the data that differs from less than a factor 2 with the
    reference dataset.
    """
    if weights is None:
        loc_weights = np.ones_like(y_ref)
    else:
        loc_weights = np.array(weights)
    
    valid_mask = y_ref > tol
    within_factor_2 = np.logical_and(y_pred[valid_mask] / y_ref[valid_mask] >= 0.5, 
                                     y_pred[valid_mask] / y_ref[valid_mask] <= 2.0)
    
    score = np.sum(loc_weights[valid_mask] * within_factor_2)
    score += np.sum(loc_weights[~valid_mask] * (y_pred[~valid_mask] <= tol))

    return score / np.sum(loc_weights)

def fms(y_pred, y_ref, iso, weights=None):
    """
    Figure of Merit in Space.
    
    Chang et Hanna. 2004. Air quality model performance evaluation.
    """
    return compute_inter_area_over_iso(y_pred, y_ref, iso, weights)/compute_union_area_over_iso(y_pred, y_ref, iso, weights)