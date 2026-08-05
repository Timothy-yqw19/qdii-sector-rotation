#!/usr/bin/env bash
# 一键入口。用法： ./run.sh <命令>
# 不带参数会打印命令清单。

set -euo pipefail
cd "$(dirname "$0")"

# 密钥从项目内的 .env 读取，不依赖 ~/.zshrc 之类的全局配置。
# 好处：换 shell（zsh/bash）不会失效，也不会把密钥散落在全局环境里。
# .env 已在 .gitignore 中，不会被提交。
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

PY="${PY:-python3}"

usage() {
  cat <<'EOF'
用法: ./run.sh <命令>

日常最常用
  ./run.sh daily      抓最新数据 + 跑回测 + 打开仪表盘   ← 平时就用这一条
  ./run.sh app        只开仪表盘（用现有缓存，不联网）

分步执行
  ./run.sh setup      安装依赖
  ./run.sh fetch      只抓数据 + 跑回测（不开仪表盘）
  ./run.sh slow       同上，但标的间隔加大（数据源限流时用）
  ./run.sh report     用现有缓存重跑回测（=最终模型：宏观维度+月频）
  ./run.sh alldims    三维全开+日频，复现被否决维度如何稀释信号

权重方案
  ./run.sh equal      等权（基线）
  ./run.sh invvol     逆波动加权（风险平价）
  ./run.sh icw        滚动IC加权（已收缩）
  ./run.sh compare    三种方案同口径横向对比

诊断实验（逐维度IC拆解后用）
  ./run.sh macro      只用宏观维度 + 月频调仓  ← 当前数据下最有希望的配置
  ./run.sh macro-d    只用宏观维度 + 日频（对照：看月频是否真的更好）
  ./run.sh nomom      砍掉动量维度（宏观+估值）

开发与验证
  ./run.sh netcheck   逐主机连通性诊断（抓数失败时先跑这个）
  ./run.sh history    历史读数复盘 + 第5档/第1档独立事件表
  ./run.sh patterns   在事件上检验事先声明的假设（含阴性对照与置换检验）
  ./run.sh pairs      多板块对交叉检验（首次需 ./run.sh pairs-fetch 补数据）
  ./run.sh pairs-fetch 补抓多板块对所需标的后再跑
  ./run.sh oos-fetch  拉取1998年起的长历史数据(样本外检验用)
  ./run.sh oos        样本外检验:2000-2007 vs 2008-2019
  ./run.sh charts     重新生成 charts/*.png
  ./run.sh pdf        重生成图表并把 REPORT.md 导出为 PDF（需 pandoc + xelatex）
  ./run.sh test       跑13项防前视自查
  ./run.sh offline    用合成数据跑通全流程（无网络时验证代码）
  ./run.sh quick      跑回测但跳过权重敏感性（最快）
  ./run.sh clean      清掉缓存与回测输出

环境变量
  PY=python              指定解释器（默认 python3）
  TIINGO_API_KEY=xxx     强烈建议：免费且已分红复权 https://www.tiingo.com/
  FRED_API_KEY=xxx       可选：启用 ALFRED 真实数据窗
EOF
}

case "${1:-}" in
  setup)   $PY -m pip install -r requirements.txt ;;
  daily)   $PY run_pipeline.py --fetch && $PY -m streamlit run app.py ;;
  fetch)   $PY run_pipeline.py --fetch ;;
  slow)    $PY run_pipeline.py --fetch --slow ;;
  report)  $PY run_pipeline.py ;;
  alldims) $PY run_pipeline.py --dims=momentum_positioning,macro_liquidity,valuation --rebalance=D ;;
  quick)   $PY run_pipeline.py --no-sens --no-compare ;;
  equal)   $PY run_pipeline.py --scheme=equal ;;
  invvol)  $PY run_pipeline.py --scheme=inverse_vol ;;
  icw)     $PY run_pipeline.py --scheme=ic_weighted ;;
  compare) $PY run_pipeline.py --no-sens ;;
  macro)   $PY run_pipeline.py --dims=macro_liquidity --rebalance=M --no-compare ;;
  macro-d) $PY run_pipeline.py --dims=macro_liquidity --no-compare ;;
  nomom)   $PY run_pipeline.py --dims=macro_liquidity,valuation --rebalance=M --no-compare ;;
  app)     $PY -m streamlit run app.py ;;
  test)    $PY tests/test_no_lookahead.py ;;
  netcheck) $PY src/data_fetcher.py --netcheck ;;
  history)     $PY explain_history.py --months=24 ;;
  patterns)    $PY find_patterns.py ;;
  pairs)       $PY multipair.py ;;
  pairs-fetch) $PY multipair.py --fetch ;;
  oos)         $PY oos_test.py ;;
  oos-fetch)   $PY oos_test.py --fetch ;;
  charts)  $PY make_charts.py ;;
  pdf)     $PY make_charts.py && pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex \
             -V geometry:margin=2.2cm -V fontsize=10.5pt \
             -V linkcolor=blue -V urlcolor=blue \
             -V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
             --toc --toc-depth=2 && echo "→ REPORT.pdf" ;;
  offline) $PY run_pipeline.py --offline ;;
  clean)   rm -f data/*.csv output/*.csv && echo "已清空 data/ 与 output/" ;;
  *)       usage ;;
esac
