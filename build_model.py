"""Rebuild the model artifact from the raw CSVs:  python build_model.py"""
import time

from recommender import Recommender

if __name__ == "__main__":
    t = time.time()
    model = Recommender.build()
    model.save()
    print(f"Built model for {len(model)} movies in {time.time() - t:.1f}s -> artifacts/model.joblib")
