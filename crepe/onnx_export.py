#!/usr/bin/env python3
"""
Script to export CREPE models to ONNX format
"""
import argparse
import os
from crepe.core import export_model_to_onnx

def main():
    print("Starting script...")
    parser = argparse.ArgumentParser(description='Export CREPE model to ONNX format')
    parser.add_argument('capacity', choices=['tiny', 'small', 'medium', 'large', 'full'],
                        help='Model capacity to export')
    parser.add_argument('-o', '--output', default=None,
                        help='Output path for the ONNX model (default: model-{capacity}.onnx)')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version (default: 13)')
    
    args = parser.parse_args()
    
    output_path = args.output
    if output_path is None:
        output_path = f"model-{args.capacity}.onnx"
    
    print(f"Exporting {args.capacity} model to {output_path}...")
    export_model_to_onnx(
        model_capacity=args.capacity,
        output_path=output_path,
        opset=args.opset
    )
    
    print("Export complete!")

if __name__ == "__main__":
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    main()