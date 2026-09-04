from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import streamlit as st


_TABLE_HTML = """
<div class="table-frame" role="region" aria-label="검색 결과 표" tabindex="0">
  <table>
    <thead><tr id="header-row"></tr></thead>
    <tbody id="table-body"></tbody>
  </table>
</div>
"""

_TABLE_CSS = """
:host {
  color: var(--st-text-color);
  font-family: var(--st-font, sans-serif);
}

.table-frame {
  max-height: 430px;
  overflow: auto;
  border: 1px solid var(--st-border-color, rgba(49, 51, 63, 0.2));
  border-radius: var(--st-border-radius, 0.5rem);
  background: var(--st-background-color);
}

table {
  width: 100%;
  min-width: 1060px;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 0.875rem;
}

th, td {
  padding: 0.55rem 0.7rem;
  border-right: 1px solid var(--st-border-color, rgba(49, 51, 63, 0.2));
  border-bottom: 1px solid var(--st-border-color, rgba(49, 51, 63, 0.2));
  text-align: left;
  white-space: nowrap;
}

th:last-child, td:last-child {
  border-right: 0;
}

thead th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: var(--st-secondary-background-color);
  font-weight: 600;
}

thead th:hover {
  background: color-mix(in srgb, var(--st-primary-color) 8%, var(--st-secondary-background-color));
}

th button {
  width: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.35rem;
  font: inherit;
  font-weight: inherit;
  white-space: nowrap;
}

th.number button {
  justify-content: flex-end;
}

.sort-marker {
  min-width: 0.8rem;
  color: var(--st-primary-color);
  font-size: 0.72rem;
}

tbody tr {
  cursor: pointer;
  outline: none;
}

tbody tr:hover, tbody tr:focus-visible, tbody tr.selected {
  background: color-mix(in srgb, var(--st-primary-color) 10%, transparent);
}

tbody tr:last-child td {
  border-bottom: 0;
}

.number {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
"""

_TABLE_JS = """
const tableStates = new WeakMap()

export default function(component) {
  const { data, parentElement, setTriggerValue } = component
  const header = parentElement.querySelector("#header-row")
  const body = parentElement.querySelector("#table-body")
  if (!header || !body) return

  const columns = Array.isArray(data?.columns) ? data.columns : []
  const rows = Array.isArray(data?.rows) ? data.rows : []
  const state = tableStates.get(parentElement) ?? { sortKey: null, direction: "asc" }
  tableStates.set(parentElement, state)

  const sortedRows = () => {
    const indexedRows = rows.map((row, index) => ({ row, index }))
    if (!state.sortKey) return indexedRows

    const column = columns.find((item) => item.key === state.sortKey)
    if (!column) return indexedRows
    const sortKey = column.sort_key ?? column.key
    const direction = state.direction === "desc" ? -1 : 1

    return indexedRows.sort((left, right) => {
      const leftValue = left.row[sortKey]
      const rightValue = right.row[sortKey]
      const leftMissing = leftValue === null || leftValue === undefined || Number.isNaN(leftValue)
      const rightMissing = rightValue === null || rightValue === undefined || Number.isNaN(rightValue)
      if (leftMissing && rightMissing) return left.index - right.index
      if (leftMissing) return 1
      if (rightMissing) return -1

      let comparison
      if (column.numeric) {
        comparison = Number(leftValue) - Number(rightValue)
      } else {
        comparison = String(leftValue).localeCompare(String(rightValue), "ko", {
          numeric: true,
          sensitivity: "base",
        })
      }
      return comparison === 0 ? left.index - right.index : comparison * direction
    })
  }

  const render = () => {
    header.replaceChildren()
    body.replaceChildren()

    for (const column of columns) {
      const cell = document.createElement("th")
      cell.scope = "col"
      if (column.numeric) cell.classList.add("number")
      const active = state.sortKey === column.key
      cell.setAttribute("aria-sort", active ? (state.direction === "asc" ? "ascending" : "descending") : "none")

      const button = document.createElement("button")
      button.type = "button"
      button.setAttribute("aria-label", `${column.label} 열 정렬`)
      const label = document.createElement("span")
      label.textContent = String(column.label ?? "")
      const marker = document.createElement("span")
      marker.className = "sort-marker"
      marker.textContent = active ? (state.direction === "asc" ? "▲" : "▼") : "↕"
      button.append(label, marker)
      button.onclick = () => {
        if (state.sortKey === column.key) {
          state.direction = state.direction === "asc" ? "desc" : "asc"
        } else {
          state.sortKey = column.key
          state.direction = "asc"
        }
        render()
      }
      cell.appendChild(button)
      header.appendChild(cell)
    }

    for (const { row } of sortedRows()) {
      const tableRow = document.createElement("tr")
      tableRow.tabIndex = 0
      tableRow.title = "더블클릭하면 아파트 상세로 이동합니다."
      tableRow.setAttribute("aria-label", `${row.complex_name ?? "아파트 단지"} 상세 열기`)

      for (const column of columns) {
        const cell = document.createElement("td")
        cell.textContent = String(row[column.key] ?? "-")
        if (column.numeric) cell.classList.add("number")
        tableRow.appendChild(cell)
      }

      tableRow.onclick = () => {
        body.querySelectorAll("tr.selected").forEach((element) => element.classList.remove("selected"))
        tableRow.classList.add("selected")
      }
      tableRow.ondblclick = () => {
        setTriggerValue("double_click", String(row.complex_id))
      }
      tableRow.onkeydown = (event) => {
        if (event.key === "Enter") {
          event.preventDefault()
          setTriggerValue("double_click", String(row.complex_id))
        }
      }
      body.appendChild(tableRow)
    }
  }

  render()
}
"""

_DOUBLE_CLICK_TABLE = st.components.v2.component(
    "double_click_table",
    html=_TABLE_HTML,
    css=_TABLE_CSS,
    js=_TABLE_JS,
)


def double_click_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[Mapping[str, Any]],
    *,
    key: str,
    on_double_click: Callable[[], None],
) -> Any:
    """표를 표시하고 행 더블클릭 시 단지 ID를 트리거로 전달한다."""
    return _DOUBLE_CLICK_TABLE(
        key=key,
        data={"rows": list(rows), "columns": list(columns)},
        width="stretch",
        height="content",
        on_double_click_change=on_double_click,
    )
