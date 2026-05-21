from __future__ import annotations

import plotly.express as px
import streamlit as st

from llm_behavior_xai.dashboard.data import (
    filter_results,
    get_surrogate_tree_plot_path,
    load_feature_group_importance,
    load_final_results,
    load_predictions,
    load_shap_importance,
    load_style_profiles,
    load_surrogate_tree_metrics,
    load_surrogate_tree_rules,
    load_xai_metrics,
)


TARGETS = ("model_key", "language")


@st.cache_data
def cached_final_results():
    return load_final_results()


@st.cache_data
def cached_metrics():
    return load_xai_metrics()


@st.cache_data
def cached_predictions(target: str, split: str):
    return load_predictions(target, split)


@st.cache_data
def cached_shap(target: str):
    return load_shap_importance(target)


@st.cache_data
def cached_feature_group_importance():
    return load_feature_group_importance()


@st.cache_data
def cached_surrogate_tree_metrics():
    return load_surrogate_tree_metrics()


@st.cache_data
def cached_surrogate_tree_rules(target: str):
    return load_surrogate_tree_rules(target)


@st.cache_data
def cached_profiles():
    return load_style_profiles()


def main() -> None:
    st.set_page_config(page_title="LLM Behavior XAI", layout="wide")
    st.title("LLM Behavior XAI")

    results = cached_final_results()
    filtered = sidebar_filters(results)
    metrics = cached_metrics()
    profiles = cached_profiles()

    overview_tab, performance_tab, shap_tab, groups_tab, tree_tab, profiles_tab, examples_tab = st.tabs(
        ["Dataset", "XAI Performance", "SHAP", "Feature Groups", "Surrogate Tree", "Style Profiles", "Responses"]
    )

    with overview_tab:
        render_dataset_overview(filtered)

    with performance_tab:
        render_performance(metrics)

    with shap_tab:
        render_shap()

    with groups_tab:
        render_feature_groups()

    with tree_tab:
        render_surrogate_tree()

    with profiles_tab:
        render_profiles(profiles)

    with examples_tab:
        render_responses(filtered)


def sidebar_filters(results):
    st.sidebar.header("Filters")
    split_options = sorted(results["split"].dropna().unique())
    splits = st.sidebar.multiselect("Split", split_options, default=split_options)
    models = st.sidebar.multiselect(
        "Model",
        sorted(results["model_key"].dropna().unique()),
        default=sorted(results["model_key"].dropna().unique()),
    )
    languages = st.sidebar.multiselect(
        "Language",
        sorted(results["language"].dropna().unique()),
        default=sorted(results["language"].dropna().unique()),
    )
    categories = st.sidebar.multiselect(
        "Category",
        sorted(results["category"].dropna().unique()),
        default=sorted(results["category"].dropna().unique()),
    )
    paraphrase_labels = {"Base prompt": False, "Paraphrase": True}
    paraphrase_selection = st.sidebar.multiselect(
        "Prompt variant",
        list(paraphrase_labels.keys()),
        default=list(paraphrase_labels.keys()),
    )
    paraphrase_values = [paraphrase_labels[label] for label in paraphrase_selection]
    return filter_results(results, splits, models, languages, paraphrase_values, categories)


def render_dataset_overview(results) -> None:
    if results.empty:
        st.info("No responses match the selected filters.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Responses", f"{len(results):,}")
    metric_cols[1].metric("Models", results["model_key"].nunique())
    metric_cols[2].metric("Prompts", results["prompt_id"].nunique())
    metric_cols[3].metric("Categories", results["category"].nunique())

    left, right = st.columns(2)
    with left:
        model_counts = results.groupby(["split", "model_key"]).size().reset_index(name="count")
        st.plotly_chart(
            px.bar(model_counts, x="model_key", y="count", color="split", barmode="group"),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            px.histogram(results, x="response_length", color="model_key", marginal="box", nbins=40),
            use_container_width=True,
        )

    st.dataframe(
        results[["split", "prompt_id", "category", "language", "is_paraphrase", "model_key", "response_length"]]
    )


def render_performance(metrics) -> None:
    if metrics.empty:
        st.info("Run `make train_xai` to generate classifier metrics.")
        return

    clean_metrics = metrics.drop(columns=["classification_report"], errors="ignore")
    st.dataframe(clean_metrics)
    st.plotly_chart(
        px.bar(clean_metrics, x="target", y="macro_f1", color="split", barmode="group", range_y=[0, 1]),
        use_container_width=True,
    )


def render_shap() -> None:
    target = st.selectbox("Target", TARGETS, key="shap_target")
    shap_df = cached_shap(target)
    if shap_df.empty:
        st.info("Run `make train_xai` to generate SHAP artifacts.")
        return

    class_names = sorted(shap_df["class_name"].dropna().unique())
    default_class_index = class_names.index("__overall__") if "__overall__" in class_names else 0
    selected_class = st.selectbox("Class", class_names, index=default_class_index)
    top_n = st.slider("Top features", 5, 30, 15)
    class_df = shap_df[shap_df["class_name"] == selected_class].sort_values("rank").head(top_n)
    st.plotly_chart(
        px.bar(class_df.sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h"),
        use_container_width=True,
    )
    st.dataframe(class_df)


def render_feature_groups() -> None:
    group_df = cached_feature_group_importance()
    if group_df.empty:
        st.info("Run the notebook to generate feature group importance.")
        return

    target = st.selectbox("Target", TARGETS, key="group_target")
    target_df = group_df[group_df["target"] == target].copy()
    methods = sorted(target_df["method"].dropna().unique())
    if not methods:
        st.info("No feature group importance is available for this target.")
        return
    method = st.selectbox("Importance method", methods, key="group_method")

    plot_df = target_df[target_df["method"] == method].sort_values("importance_share")
    st.plotly_chart(
        px.bar(
            plot_df,
            x="importance_share",
            y="feature_group",
            orientation="h",
            labels={"importance_share": "Share of importance", "feature_group": "Feature group"},
        ),
        use_container_width=True,
    )
    st.dataframe(plot_df.sort_values("importance_share", ascending=False))


def render_surrogate_tree() -> None:
    metrics = cached_surrogate_tree_metrics()
    if metrics.empty:
        st.info("Run the notebook to generate surrogate decision trees.")
        return

    target = st.selectbox("Target", TARGETS, key="surrogate_target")
    target_metrics = metrics[metrics["target"] == target].copy()
    st.dataframe(target_metrics)

    plot_path = get_surrogate_tree_plot_path(target)
    if plot_path.exists():
        st.image(str(plot_path))

    rules = cached_surrogate_tree_rules(target)
    if rules:
        st.code(rules, language="text")


def render_profiles(profiles) -> None:
    top_features = profiles["top_features"]
    sensitivity = profiles["sensitivity"]

    if top_features.empty:
        st.info("Run `make build_profiles` to generate style profiles.")
        return

    model = st.selectbox("Model", sorted(top_features["model_key"].unique()))
    model_features = top_features[top_features["model_key"] == model]
    st.plotly_chart(
        px.bar(
            model_features.sort_values("effect_size"),
            x="effect_size",
            y="feature",
            color="direction",
            orientation="h",
        ),
        use_container_width=True,
    )
    st.dataframe(model_features)

    if not sensitivity.empty:
        st.subheader("Language and paraphrase sensitivity")
        st.dataframe(sensitivity[sensitivity["model_key"] == model])


def render_responses(results) -> None:
    if results.empty:
        st.info("No responses match the selected filters.")
        return

    target = st.selectbox("Prediction target", TARGETS, key="prediction_target")
    split = st.selectbox("Prediction split", sorted(results["split"].dropna().unique()), key="prediction_split")
    predictions = cached_predictions(target, split)

    left, right = st.columns(2)
    with left:
        st.subheader("Filtered response examples")
        example_rows = results.sort_values(["split", "prompt_id", "model_key"]).head(20)
        for _, row in example_rows.iterrows():
            with st.expander(f"{row['split']} | {row['prompt_id']} | {row['model_key']}"):
                st.write(row.get("prompt_text", ""))
                st.write(row.get("response", ""))

    with right:
        st.subheader("Misclassified cases")
        if predictions.empty:
            st.info("Run `make train_xai` to generate predictions.")
            return
        correct_mask = predictions["correct"].astype(str).str.lower() == "true"
        misses = predictions[~correct_mask].head(20)
        if misses.empty:
            st.success("No misclassified cases for this target and split.")
            return
        for _, row in misses.iterrows():
            with st.expander(f"{row.get('prompt_id', '')} | true: {row['y_true']} | predicted: {row['y_pred']}"):
                st.write(row.get("prompt_text", ""))
                st.write(row.get("response", ""))


if __name__ == "__main__":
    main()
