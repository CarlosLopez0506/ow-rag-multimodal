"""Gradio web interface for the OW RAG recommender."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

import gradio as gr

from .data import load_heroes
from .history import DEFAULT_HISTORY_PATH, clear_history, load_history, record_played, top_played_slugs
from .recommender import (
    DEFAULT_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    OWRAGMultimodalRecommender,
)


def _history_markdown() -> str:
    """Builds a markdown summary for the persisted play history.

    Returns:
        Markdown content for the history panel.
    """

    history = load_history(DEFAULT_HISTORY_PATH)
    counts = history.get("played_counts", {})
    if not isinstance(counts, dict) or not counts:
        return "### Tus héroes frecuentes\nAún no tienes historial."

    ranking = sorted(
        ((str(slug), int(count)) for slug, count in counts.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:15]
    lines = ["### Tus héroes frecuentes", "Tus héroes más jugados:"]
    for idx, (slug, count) in enumerate(ranking, start=1):
        lines.append(f"{idx}. `{slug}` - {count} veces")
    return "\n".join(lines)


def _is_port_available(host: str, port: int) -> bool:
    """Checks whether a TCP port can be bound on a host.

    Args:
        host: Hostname or IP to bind.
        port: TCP port number.

    Returns:
        ``True`` when binding succeeds, otherwise ``False``.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _pick_available_port(host: str, start_port: int, max_tries: int) -> int:
    """Finds an available port in a consecutive range.

    Args:
        host: Hostname or IP to bind.
        start_port: First port to try.
        max_tries: Number of consecutive ports to probe.

    Returns:
        First available port in the checked range.

    Raises:
        OSError: If no port is available in the requested range.
    """

    tries = max(1, max_tries)
    for offset in range(tries):
        candidate = start_port + offset
        if _is_port_available(host, candidate):
            return candidate
    raise OSError(
        f"No hay puertos libres entre {start_port} y {start_port + tries - 1}. "
        "Usa --port con otro valor."
    )


def build_interface(
    heroes_path: Path = Path("data/heroes.json"),
    images_dir: Path = Path("data/images"),
) -> gr.Blocks:
    """Builds the full Gradio app with stateful callbacks.

    Args:
        heroes_path: Path to the heroes dataset.

    Returns:
        A configured ``gr.Blocks`` application.
    """

    heroes = load_heroes(heroes_path)
    slug_to_name = {hero.slug: hero.name for hero in heroes}
    label_to_slug = {f"{hero.name} [{hero.slug}]": hero.slug for hero in heroes}
    hero_labels = sorted(label_to_slug.keys())

    recommender_cache: OWRAGMultimodalRecommender | None = None

    def build_effective_refs(
        selected_slugs: list[str], include_history: bool, history_top_n: int
    ) -> list[str]:
        """Combines UI selection and top history into effective played refs.

        Args:
            selected_slugs: Slugs explicitly selected in the UI.
            include_history: Whether to append historical top picks.
            history_top_n: Maximum number of historical slugs to include.

        Returns:
            Ordered list of effective hero references without duplicates.
        """

        effective = list(selected_slugs or [])
        if include_history:
            for slug in top_played_slugs(limit=int(history_top_n), path=DEFAULT_HISTORY_PATH):
                if slug not in effective:
                    effective.append(slug)
        return effective

    def profile_input_markdown(
        selected_slugs: list[str], include_history: bool, history_top_n: int
    ) -> str:
        """Renders markdown describing effective profile inputs.

        Args:
            selected_slugs: Slugs explicitly selected in the UI.
            include_history: Whether history is considered.
            history_top_n: Number of historical items considered.

        Returns:
            Markdown content for the profile-input panel.
        """

        selected = list(selected_slugs or [])
        effective = build_effective_refs(selected, include_history, history_top_n)
        if not effective:
            return "### Tu selección\nSelecciona héroes arriba o describe tu estilo de juego."

        selected_set = set(selected)
        selected_count = sum(1 for slug in effective if slug in selected_set)
        history_count = len(effective) - selected_count
        lines = [
            "### Tu selección",
            f"{len(effective)} héroes en total — {selected_count} seleccionados, {history_count} de tu historial",
        ]
        for idx, slug in enumerate(effective, start=1):
            source = "seleccionado" if slug in selected_set else "frecuente"
            lines.append(f"{idx}. {slug_to_name.get(slug, slug)} — {source}")
        return "\n".join(lines)

    def sync_selected(
        hero_labels_selected: list[str] | None,
        prev_selected_slugs: list[str],
        include_history: bool,
        history_top_n: int,
    ) -> tuple[list[str], str, str, str]:
        """Synchronizes selected labels into slugs and updates history/UI state.

        Args:
            hero_labels_selected: UI label selections.
            prev_selected_slugs: Previous selected slugs from state.
            include_history: Whether history is considered in profile preview.
            history_top_n: Number of historical items considered.

        Returns:
            Tuple with updated selected slugs, profile markdown, history markdown,
            and status message.
        """

        selected_labels = list(hero_labels_selected or [])
        new_slugs: list[str] = []
        for label in selected_labels:
            slug = label_to_slug.get(label)
            if slug and slug not in new_slugs:
                new_slugs.append(slug)

        previous = list(prev_selected_slugs or [])
        added = [slug for slug in new_slugs if slug not in previous]
        if added:
            record_played(added, DEFAULT_HISTORY_PATH)
            status = f"{len(added)} héroe(s) guardados en tu historial."
        else:
            status = "Selección actualizada."

        return (
            new_slugs,
            profile_input_markdown(new_slugs, include_history, history_top_n),
            _history_markdown(),
            status,
        )

    def clear_history_state(
        selected_slugs: list[str], include_history: bool, history_top_n: int
    ) -> tuple[str, str, str]:
        """Clears persisted history and refreshes dependent UI panels.

        Args:
            selected_slugs: Current selected slugs.
            include_history: Whether history is considered in profile preview.
            history_top_n: Number of historical items considered.

        Returns:
            Tuple with history markdown, profile markdown, and status message.
        """

        clear_history(DEFAULT_HISTORY_PATH)
        current = list(selected_slugs or [])
        return (
            _history_markdown(),
            profile_input_markdown(current, include_history, history_top_n),
            "Historial borrado.",
        )

    def refresh_effective(
        selected_slugs: list[str], include_history: bool, history_top_n: int
    ) -> tuple[str, str]:
        """Refreshes profile and history markdown panels.

        Args:
            selected_slugs: Current selected slugs.
            include_history: Whether history is considered.
            history_top_n: Number of historical items considered.

        Returns:
            Tuple with profile markdown and history markdown.
        """

        current = list(selected_slugs or [])
        return profile_input_markdown(current, include_history, history_top_n), _history_markdown()

    def recommend_from_ui(
        query: str,
        selected_slugs: list[str],
        role: str,
        top_k: int,
        include_history: bool,
        history_top_n: int,
        refresh_cache: bool,
        show_context: bool,
        alpha_image: float,
    ) -> tuple[str, str, str, str, str, list]:
        """Runs recommendation flow from UI inputs and formats output panels.

        Args:
            query: Free-text query entered by the user.
            selected_slugs: Currently selected hero slugs.
            role: Optional role filter.
            top_k: Number of recommendations to request.
            include_history: Whether to include persisted history in profile inputs.
            history_top_n: Number of top historical heroes to include.
            refresh_cache: Whether to force rebuilding embedding cache.
            show_context: Whether to expose retrieved context in profile output.

        Returns:
            Tuple with recommendation markdown, profile markdown, profile-input markdown,
            history markdown, and status message.
        """

        nonlocal recommender_cache
        played_refs = build_effective_refs(
            selected_slugs=list(selected_slugs or []),
            include_history=include_history,
            history_top_n=int(history_top_n),
        )

        if not query.strip() and not played_refs:
            return (
                "Selecciona al menos un héroe o describe cómo te gusta jugar.",
                "",
                profile_input_markdown(selected_slugs or [], include_history, history_top_n),
                _history_markdown(),
                "Faltan datos para recomendar.",
                [],
            )

        if refresh_cache or recommender_cache is None:
            recommender_cache = OWRAGMultimodalRecommender(
                heroes_path=heroes_path,
                cache_dir=DEFAULT_CACHE_DIR,
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                force_refresh_cache=refresh_cache,
            )

        result = recommender_cache.recommend(
            query=query,
            played_refs=played_refs,
            top_k=int(top_k),
            role_filter=role or None,
            profile_top_k=6,
            exclude_played=True,
            alpha_image=float(alpha_image),
        )

        rec_images: list[tuple[str, str]] = []
        if not result.recommendations:
            rec_md = "### Héroes recomendados\nNo se encontraron resultados."
        else:
            lines = ["### Héroes recomendados"]
            for idx, rec in enumerate(result.recommendations, start=1):
                lines.append(f"{idx}. **{rec.name}** — {rec.role} ({rec.score}% de compatibilidad)")
                img_path = images_dir / f"{rec.slug}.png"
                if img_path.exists():
                    rec_images.append((str(img_path), f"{rec.name} · {rec.role} · {rec.score}%"))
            rec_md = "\n".join(lines)

        profile_md = ""
        if result.profile:
            lines = ["### ¿Por qué estas recomendaciones?"]
            lines.append(f"**Basado en:** {', '.join(result.profile.played_heroes)}")
            if result.profile.dominant_roles:
                lines.append(f"**Rol principal:** {', '.join(result.profile.dominant_roles)}")
            if result.profile.signature_traits:
                lines.append(f"**Estilo detectado:** {', '.join(result.profile.signature_traits[:8])}")
            if show_context and result.profile.retrieved_context:
                lines.append("")
                lines.append("**Héroes similares analizados:**")
                for i, ctx in enumerate(result.profile.retrieved_context, start=1):
                    lines.append(f"{i}. {ctx.name} — {ctx.role} ({ctx.score}% similitud)")
            profile_md = "\n".join(lines)

        return (
            rec_md,
            profile_md,
            profile_input_markdown(selected_slugs or [], include_history, history_top_n),
            _history_markdown(),
            "¡Listo! Aquí están tus recomendaciones.",
            rec_images,
        )

    with gr.Blocks(title="Recomendador de Héroes · Overwatch") as demo:
        gr.Markdown("## Recomendador de Héroes · Overwatch")
        gr.Markdown(
            "Selecciona los héroes que ya juegas, describe cómo te gusta jugar y obtén recomendaciones personalizadas."
        )

        selected_state = gr.State([])
        status_box = gr.Markdown("Listo para recomendar.")

        hero_choice = gr.Dropdown(
            choices=hero_labels,
            multiselect=True,
            label="Héroes que ya juegas",
            info="Se guardan automáticamente en tu historial.",
        )
        query = gr.Textbox(
            label="¿Cómo te gusta jugar?",
            placeholder="Ejemplo: me gustan los dives rápidos, presión constante y picks aislados",
        )

        run_btn = gr.Button("Recomendar", variant="primary")

        profile_input_box = gr.Markdown(profile_input_markdown([], True, 8))

        with gr.Accordion("Opciones", open=False):
            with gr.Row():
                include_history = gr.Checkbox(value=True, label="Recordar mis héroes frecuentes")
                history_top_n = gr.Slider(1, 15, value=8, step=1, label="Cuántos héroes frecuentes considerar")
                role = gr.Dropdown(choices=["", "Tank", "Damage", "Support"], value="", label="Filtrar por rol")
                top_k = gr.Slider(1, 10, value=5, step=1, label="Número de recomendaciones")

            with gr.Row():
                alpha_image = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Peso del análisis visual", info="Qué tanto influyen las imágenes de los héroes en la recomendación.")
                refresh_cache = gr.Checkbox(value=False, label="Reiniciar análisis desde cero")
                show_context = gr.Checkbox(value=True, label="Ver detalle del análisis")

            clear_history_btn = gr.Button("Borrar historial")
            history_box = gr.Markdown(_history_markdown())

        rec_out = gr.Markdown()
        rec_gallery = gr.Gallery(
            label="Imágenes",
            columns=5,
            height="auto",
            object_fit="contain",
        )
        profile_out = gr.Markdown()

        hero_choice.change(
            fn=sync_selected,
            inputs=[hero_choice, selected_state, include_history, history_top_n],
            outputs=[selected_state, profile_input_box, history_box, status_box],
        )
        clear_history_btn.click(
            fn=clear_history_state,
            inputs=[selected_state, include_history, history_top_n],
            outputs=[history_box, profile_input_box, status_box],
        )
        include_history.change(
            fn=refresh_effective,
            inputs=[selected_state, include_history, history_top_n],
            outputs=[profile_input_box, history_box],
        )
        history_top_n.change(
            fn=refresh_effective,
            inputs=[selected_state, include_history, history_top_n],
            outputs=[profile_input_box, history_box],
        )
        run_btn.click(
            fn=recommend_from_ui,
            inputs=[
                query,
                selected_state,
                role,
                top_k,
                include_history,
                history_top_n,
                refresh_cache,
                show_context,
                alpha_image,
            ],
            outputs=[rec_out, profile_out, profile_input_box, history_box, status_box, rec_gallery],
        )
        query.submit(
            fn=recommend_from_ui,
            inputs=[
                query,
                selected_state,
                role,
                top_k,
                include_history,
                history_top_n,
                refresh_cache,
                show_context,
                alpha_image,
            ],
            outputs=[rec_out, profile_out, profile_input_box, history_box, status_box, rec_gallery],
        )

    return demo


def main() -> int:
    """Parses UI arguments and launches the Gradio application.

    Returns:
        Process-like exit code where ``0`` indicates successful startup.
    """

    parser = argparse.ArgumentParser(description="Mini interfaz para OW RAG Multimodal.")
    parser.add_argument("--host", default="127.0.0.1", help="Host de la UI")
    parser.add_argument("--port", type=int, default=7860, help="Puerto de la UI")
    parser.add_argument(
        "--max-port-tries",
        type=int,
        default=30,
        help="Cantidad de puertos consecutivos a intentar si el puerto base está ocupado.",
    )
    parser.add_argument("--share", action="store_true", help="Habilita URL pública temporal")
    parser.add_argument(
        "--heroes-path",
        type=Path,
        default=Path("data/heroes.json"),
        help="Ruta al dataset de héroes",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/images"),
        help="Directorio con los retratos PNG de héroes",
    )
    args = parser.parse_args()

    port = _pick_available_port(args.host, args.port, args.max_port_tries)
    if port != args.port:
        print(f"Puerto {args.port} ocupado. Usando puerto {port}.")

    app = build_interface(heroes_path=args.heroes_path, images_dir=args.images_dir)
    app.launch(server_name=args.host, server_port=port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
