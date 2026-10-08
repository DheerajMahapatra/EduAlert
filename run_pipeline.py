"""Run the whole EduAlert pipeline end-to-end:  python run_pipeline.py [--mlflow] [--n 6000]"""
import argparse

from src import association, clustering, data_generation, data_preprocessing, eda, make_report, train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="number of synthetic students")
    ap.add_argument("--mlflow", action="store_true", help="log runs to MLflow")
    a = ap.parse_args()
    data_generation.main(a.n)
    data_preprocessing.main()
    eda.run()
    train.main(a.mlflow)
    clustering.main()
    association.main()
    make_report.main()
    print("\nDone. Launch the dashboard with:  streamlit run app/app.py")
