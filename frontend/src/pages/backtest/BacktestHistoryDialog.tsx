import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { History, Loader2, Trash2, X } from 'lucide-react'
import { Modal } from '@/components/Modal'
import { toast } from '@/components/Toast'
import { api, type BacktestHistoryDetail, type BacktestHistoryItem } from '@/lib/api'
import { cnDateTimeFromUtc, fmtPct, priceColorClass } from '@/lib/format'
import { QK } from '@/lib/queryKeys'

function metric(item: BacktestHistoryItem, key: string): number | null {
  const value = item.metrics[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function fmtSharpe(item: BacktestHistoryItem): string {
  const value = metric(item, 'sharpe')
  return value == null ? '—' : value.toFixed(2)
}

function fmtTrades(item: BacktestHistoryItem): string {
  const value = metric(item, 'n_trades')
  return value == null ? '—' : String(Math.round(value))
}

function fmtRange(item: BacktestHistoryItem): string {
  const start = item.start.slice(0, 10)
  const end = item.end.slice(0, 10)
  if (!start && !end) return '—'
  return `${start || '—'} ~ ${end || '—'}`
}

function ReturnCell({ value }: { value: number | null }) {
  return (
    <span className={priceColorClass(value)}>{value == null ? '—' : fmtPct(value)}</span>
  )
}

export function BacktestHistoryDialog({ onClose, onLoad }: {
  onClose: () => void
  /** 查看某次回测: 把完整结果回填到回测页 (由回测页接线) */
  onLoad: (detail: BacktestHistoryDetail) => void
}) {
  const queryClient = useQueryClient()
  const history = useQuery({ queryKey: QK.backtestHistory, queryFn: api.backtestHistory })
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [loadingId, setLoadingId] = useState<string | null>(null)
  const items = useMemo(() => history.data?.items ?? [], [history.data])

  const remove = useMutation({
    mutationFn: async (ids: string[]) => {
      for (const id of ids) await api.backtestHistoryDelete(id)
      return ids.length
    },
    onSuccess: count => toast(`已删除 ${count} 条回测记录`, 'success'),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: QK.backtestHistory })
      setSelected(new Set())
    },
  })

  const toggle = (id: string) => {
    setSelected(current => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }
  const allSelected = items.length > 0 && selected.size === items.length
  const toggleAll = () => {
    setSelected(allSelected ? new Set() : new Set(items.map(item => item.id)))
  }

  const deleteSelected = () => {
    const ids = Array.from(selected)
    if (ids.length === 0) return
    if (!window.confirm(`确认删除选中的 ${ids.length} 条回测记录？删除后不可恢复。`)) return
    remove.mutate(ids)
  }

  const openDetail = async (item: BacktestHistoryItem) => {
    if (!item.has_result || loadingId) return
    setLoadingId(item.id)
    try {
      const detail = await api.backtestHistoryGet(item.id)
      if (!detail.result) {
        toast('该记录只保留了指标摘要，完整结果不可用', 'error')
        return
      }
      onLoad(detail)
    } catch {
      // request() 已统一弹出错误提示
    } finally {
      setLoadingId(null)
    }
  }

  return (
    <Modal
      onClose={onClose}
      labelledBy="backtest-history-title"
      panelClassName="flex max-h-[82vh] w-[96vw] max-w-6xl flex-col overflow-hidden rounded-card border border-border bg-base shadow-2xl"
    >
      <header className="flex shrink-0 items-center gap-3 border-b border-border px-4 py-3">
        <History className="h-4 w-4 text-accent" />
        <div className="min-w-0">
          <h2 id="backtest-history-title" className="text-sm font-semibold text-foreground">回测历史</h2>
          <div className="mt-0.5 text-[11px] text-muted">
            每次回测成功后自动保存，重启程序后仍可对比查看；最多保留最近 100 条
          </div>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="ml-auto flex h-8 w-8 items-center justify-center rounded-btn text-muted transition-colors hover:bg-elevated hover:text-foreground"
          title="关闭"
          aria-label="关闭回测历史"
        >
          <X className="h-4 w-4" />
        </button>
      </header>

      <div className="flex shrink-0 items-center gap-3 border-b border-border bg-surface/50 px-4 py-2">
        <span className="text-[11px] text-muted">{items.length} 条记录</span>
        {selected.size > 0 && (
          <span className="text-[11px] text-accent">已选 {selected.size} 条</span>
        )}
        <button
          type="button"
          onClick={deleteSelected}
          disabled={selected.size === 0 || remove.isPending}
          className="ml-auto inline-flex h-8 items-center gap-1.5 rounded-btn border border-danger/30 px-2.5 text-[11px] font-medium text-danger transition-colors hover:bg-danger/10 disabled:cursor-not-allowed disabled:border-border disabled:text-muted disabled:opacity-50"
        >
          {remove.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
          删除选中
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {history.isLoading && (
          <div className="px-4 py-10 text-center text-sm text-muted">加载中…</div>
        )}
        {history.isError && (
          <div className="m-4 rounded-btn border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
            {String((history.error as Error).message)}
          </div>
        )}
        {!history.isLoading && !history.isError && items.length === 0 && (
          <div className="px-4 py-12 text-center">
            <History className="mx-auto h-7 w-7 text-muted/50" />
            <div className="mt-3 text-sm font-medium text-secondary">暂无回测记录</div>
            <div className="mt-1 text-[11px] text-muted">回测成功后会自动出现在这里</div>
          </div>
        )}
        {items.length > 0 && (
          <table className="w-full min-w-[880px] border-collapse text-xs">
            <thead className="sticky top-0 z-10 bg-surface text-[11px] text-muted">
              <tr className="border-b border-border">
                <th className="w-9 px-3 py-2">
                  <input
                    type="checkbox"
                    checked={allSelected}
                    onChange={toggleAll}
                    aria-label={allSelected ? '取消全选' : '全选'}
                    className="h-3.5 w-3.5 accent-accent"
                  />
                </th>
                <th className="px-2 py-2 text-left font-medium">回测时间</th>
                <th className="px-2 py-2 text-left font-medium">策略</th>
                <th className="px-2 py-2 text-left font-medium">区间</th>
                <th className="px-2 py-2 text-right font-medium">总收益</th>
                <th className="px-2 py-2 text-right font-medium">年化</th>
                <th className="px-2 py-2 text-right font-medium">最大回撤</th>
                <th className="px-2 py-2 text-right font-medium">夏普</th>
                <th className="px-2 py-2 text-right font-medium">胜率</th>
                <th className="px-2 py-2 text-right font-medium">交易数</th>
                <th className="w-16 px-2 py-2 text-right font-medium">操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map(item => {
                const checking = selected.has(item.id)
                const loading = loadingId === item.id
                return (
                  <tr
                    key={item.id}
                    className={`border-b border-border/60 transition-colors hover:bg-elevated/40 ${checking ? 'bg-accent/5' : ''}`}
                  >
                    <td className="px-3 py-2">
                      <input
                        type="checkbox"
                        checked={checking}
                        onChange={() => toggle(item.id)}
                        aria-label={`选择 ${item.strategy_name} ${item.created_at}`}
                        className="h-3.5 w-3.5 accent-accent"
                      />
                    </td>
                    <td className="whitespace-nowrap px-2 py-2 text-secondary">
                      {cnDateTimeFromUtc(item.created_at) || '—'}
                    </td>
                    <td className="max-w-[14rem] px-2 py-2">
                      <div className="truncate font-medium text-foreground" title={item.strategy_name}>
                        {item.strategy_name}
                      </div>
                      <div className="truncate text-[10px] text-muted">{item.strategy_id}</div>
                    </td>
                    <td className="whitespace-nowrap px-2 py-2 text-muted">{fmtRange(item)}</td>
                    <td className="px-2 py-2 text-right tabular-nums">
                      <ReturnCell value={metric(item, 'total_return')} />
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums">
                      <ReturnCell value={metric(item, 'annual_return')} />
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums text-secondary">
                      {fmtPct(metric(item, 'max_drawdown'))}
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums text-secondary">{fmtSharpe(item)}</td>
                    <td className="px-2 py-2 text-right tabular-nums text-secondary">
                      {fmtPct(metric(item, 'win_rate'))}
                    </td>
                    <td className="px-2 py-2 text-right tabular-nums text-secondary">{fmtTrades(item)}</td>
                    <td className="px-2 py-2 text-right">
                      <button
                        type="button"
                        onClick={() => openDetail(item)}
                        disabled={!item.has_result || loading || !!loadingId}
                        title={item.has_result ? '查看这次回测的完整结果' : '结果过大，仅保留指标摘要'}
                        className="inline-flex h-7 items-center gap-1 rounded-btn px-2 text-[11px] text-accent transition-colors hover:bg-accent/10 disabled:cursor-not-allowed disabled:text-muted disabled:opacity-50"
                      >
                        {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
                        查看
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </Modal>
  )
}
