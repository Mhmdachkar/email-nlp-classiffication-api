#!/usr/bin/env python3
"""
📊 TRAINING DATA GENERATOR
==========================
Generate high-quality, long-format email training data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from components.enhanced_dataset_generator import EnhancedEmailDatasetGenerator
from datetime import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate email training dataset")
    parser.add_argument("--samples-per-category", type=int, default=500,
                       help="Number of samples per category")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output CSV file name")
    
    args = parser.parse_args()
    
    print("🎯 GENERATING EMAIL TRAINING DATASET")
    print("=" * 50)
    
    # Create generator
    generator = EnhancedEmailDatasetGenerator()
    
    # Generate dataset
    df = generator.generate_comprehensive_dataset(
        spam_count=args.samples_per_category,
        work_count=args.samples_per_category,
        personal_count=args.samples_per_category,
        urgent_count=args.samples_per_category,
        standard_count=args.samples_per_category
    )
    
    # Save dataset
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"email_training_dataset_{timestamp}.csv"
    
    df.to_csv(args.output_file, index=False)
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📁 Saved to: {args.output_file}")
    print(f"📊 Total samples: {len(df):,}")
    
    # Show sample from each category
    print(f"\n📖 SAMPLE EMAILS")
    print("=" * 50)
    
    for category in df['category'].unique():
        sample = df[df['category'] == category].iloc[0]['email']
        print(f"\n{category.upper()} SAMPLE:")
        print("-" * 30)
        # Show first 300 chars
        print(sample[:300] + "..." if len(sample) > 300 else sample)
    
    return 0

if __name__ == "__main__":
    exit(main())
