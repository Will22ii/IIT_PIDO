# DOE/executor/doe_report_builder.py

from __future__ import annotations

import pandas as pd


class DOEReportBuilder:

    @staticmethod
    def build(
        *,
        problem_name: str,
        workflow_info: dict,
        results: list,
        n_samples: int,
        dimension: int,
        objective_sense: str = "min",
        dump_table: bool = False,   # ⭐ optional
    ) -> list[str]:
        lines: list[str] = []

        # -----------------------------
        # Problem
        # -----------------------------
        lines.append("[Problem]")
        lines.append(f"- Name          : {problem_name}")
        lines.append(f"- DOE Algorithm : {workflow_info.get('DOE')}")
        lines.append(f"- MODELER       : {workflow_info.get('MODELER')}")
        lines.append(f"- EXPLORER      : {workflow_info.get('EXPLORER')}")
        lines.append(f"- OPT           : {workflow_info.get('OPT')}")
        lines.append(f"- Samples       : {n_samples}")
        lines.append(f"- Dimension     : {dimension}")
        lines.append("")

        # -----------------------------
        # Best feasible (objective_sense 기준)
        # -----------------------------
        feasible = [r for r in results if r.get("feasible", False)]
        sense = str(objective_sense or "min").strip().lower()
        if feasible:
            if sense == "max":
                best = max(feasible, key=lambda r: r["objective"])
            else:
                best = min(feasible, key=lambda r: r["objective"])
        else:
            best = None

        lines.append("----------------------------------------")
        lines.append("[Best Feasible Result]")
        lines.append("----------------------------------------")

        if best is None:
            lines.append("No feasible solution found.")
        else:
            lines.append(f"ID        : {best['id']}")
            lines.append(f"Objective : {best['objective']}")
            lines.append(f"x         : {best['x']}")
            constraints = best.get("constraints") or {}
            for cname, cinfo in constraints.items():
                try:
                    val = cinfo.get("value", cinfo)
                except Exception:
                    val = cinfo
                lines.append(f"{cname:12}: {val}")
            lines.append(f"Feasible  : {best.get('feasible')}")
            lines.append(f"Success   : {best.get('success')}")

        lines.append("")

        # -----------------------------
        # All results (row dump, optional)
        # -----------------------------
        if dump_table:
            lines.append("----------------------------------------")
            lines.append("[All Results – Full Table]")
            lines.append("----------------------------------------")

            if not results:
                lines.append("(empty)")
                return lines

            # results -> DataFrame
            df = pd.DataFrame(results)

            # expand x -> x_0, x_1, ...
            if "x" in df.columns:
                x_df = pd.DataFrame(df["x"].tolist())
                x_df.columns = [f"x_{i}" for i in range(x_df.shape[1])]
                df = pd.concat([df.drop(columns=["x"]), x_df], axis=1)

            # 안정적인 컬럼 순서 (존재하는 것만)
            preferred_cols = [
                "id",
                "source",
                "round",
                "feasible",
                "success",
                "objective",
            ]
            x_cols = [c for c in df.columns if c.startswith("x_")]
            other_cols = [
                c for c in df.columns
                if c not in preferred_cols and c not in x_cols
            ]

            ordered_cols = (
                [c for c in preferred_cols if c in df.columns]
                + sorted(x_cols, key=lambda s: int(s.split("_")[1]))
                + other_cols
            )
            df = df[ordered_cols]

            # dump as CSV text (index=False)
            table_str = df.to_csv(index=False)
            lines.extend(table_str.strip().splitlines())

        else:
            # -----------------------------
            # All results (summary, 기존 유지)
            # -----------------------------
            lines.append("----------------------------------------")
            lines.append("[All Results]")
            lines.append("----------------------------------------")
            lines.append(
                f"- feasible_count: {sum(bool(r.get('feasible')) for r in results)}"
            )
            lines.append(
                f"- success_count : {sum(bool(r.get('success')) for r in results)}"
            )

        return lines
