"""
Test file for moving_average.py
Tests moving average functions and class
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml_code.moving_average import moving_average, MovingAverage
import pytest


def test_moving_average_basic():
    """Test basic moving average calculation"""
    arr = [1, 2, 3, 4, 5]
    result = moving_average(arr, 3)
    # Window of 3: [(1+2+3)/3, (2+3+4)/3, (3+4+5)/3] = [2.0, 3.0, 4.0]
    assert result == [2.0, 3.0, 4.0]


def test_moving_average_window_size_one():
    """Test moving average with window size 1"""
    arr = [1, 2, 3, 4, 5]
    result = moving_average(arr, 1)
    assert result == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_moving_average_window_size_equal_length():
    """Test moving average with window size equal to array length"""
    arr = [1, 2, 3, 4, 5]
    result = moving_average(arr, 5)
    # Single window: (1+2+3+4+5)/5 = 3.0
    assert result == [3.0]


def test_moving_average_window_size_greater_than_length():
    """Test moving average with window size greater than array length"""
    arr = [1, 2, 3]
    result = moving_average(arr, 5)
    # No windows can be formed
    assert result == []


def test_moving_average_empty_array():
    """Test moving average with empty array"""
    arr = []
    result = moving_average(arr, 3)
    assert result == []


def test_moving_average_negative_numbers():
    """Test moving average with negative numbers"""
    arr = [-1, -2, -3, -4]
    result = moving_average(arr, 2)
    # [(-1-2)/2, (-2-3)/2, (-3-4)/2] = [-1.5, -2.5, -3.5]
    assert result == [-1.5, -2.5, -3.5]


def test_moving_average_single_element():
    """Test moving average with single element array"""
    arr = [5]
    result = moving_average(arr, 1)
    assert result == [5.0]


def test_moving_average_class_initialization():
    """Test MovingAverage class initialization"""
    ma = MovingAverage(3)
    assert ma.window_size == 3
    assert len(ma.queue) == 0
    assert ma.window_sum == 0


def test_moving_average_class_basic():
    """Test MovingAverage class basic functionality"""
    ma = MovingAverage(3)
    
    result1 = ma.next(1)
    assert result1 == 1.0  # [1] / 1 = 1.0
    
    result2 = ma.next(10)
    assert abs(result2 - 5.5) < 1e-6  # (1 + 10) / 2 = 5.5
    
    result3 = ma.next(3)
    assert abs(result3 - 14/3) < 1e-6  # (1 + 10 + 3) / 3 ≈ 4.667


def test_moving_average_class_window_full():
    """Test MovingAverage when window is full"""
    ma = MovingAverage(3)
    
    ma.next(1)   # [1] -> avg = 1.0
    ma.next(10)  # [1, 10] -> avg = 5.5
    ma.next(3)   # [1, 10, 3] -> avg = 14/3 ≈ 4.667
    result4 = ma.next(5)   # [10, 3, 5] -> avg = (10 + 3 + 5)/3 = 6.0
    
    assert abs(result4 - 6.0) < 1e-6


def test_moving_average_class_sliding_window():
    """Test that MovingAverage correctly slides the window"""
    ma = MovingAverage(3)
    
    # Add values: [1]
    val1 = ma.next(1)
    
    # Add values: [1, 2]
    val2 = ma.next(2)
    
    # Add values: [1, 2, 3] -> full window
    val3 = ma.next(3)
    
    # Add values: [2, 3, 4] -> should remove 1
    val4 = ma.next(4)
    
    # Verify sliding window
    assert abs(val3 - 2.0) < 1e-6  # (1+2+3)/3 = 2.0
    assert abs(val4 - 3.0) < 1e-6  # (2+3+4)/3 = 3.0


def test_moving_average_class_window_size_one():
    """Test MovingAverage with window size 1"""
    ma = MovingAverage(1)
    
    assert abs(ma.next(5) - 5.0) < 1e-6
    assert abs(ma.next(10) - 10.0) < 1e-6
    assert abs(ma.next(15) - 15.0) < 1e-6


def test_moving_average_class_zero_values():
    """Test MovingAverage with zero values"""
    ma = MovingAverage(3)
    
    ma.next(0)
    ma.next(0)
    result = ma.next(0)
    
    assert result == 0.0


def test_moving_average_class_mixed_positive_negative():
    """Test MovingAverage with mixed positive and negative values"""
    ma = MovingAverage(2)
    
    result1 = ma.next(5)   # [5]
    result2 = ma.next(-3)  # [5, -3] -> (5-3)/2 = 1.0
    
    assert abs(result1 - 5.0) < 1e-6
    assert abs(result2 - 1.0) < 1e-6

