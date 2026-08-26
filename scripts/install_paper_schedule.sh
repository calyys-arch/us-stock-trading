#!/bin/bash
# Install (or remove) the launchd agent that advances the paper ledger daily.
#
#   bash scripts/install_paper_schedule.sh install
#   bash scripts/install_paper_schedule.sh status
#   bash scripts/install_paper_schedule.sh uninstall
#
# Runs weekdays at 07:00 local. US equities close at 16:00 ET, which is 04:00
# the next morning in UTC+8, so 07:00 local is comfortably after the close with
# room for Yahoo to publish. The exact hour barely matters: the ledger fills in
# whatever days it missed, so a laptop that was asleep catches up on waking.
#
# RunAtLoad is deliberately false. Loading the agent should not silently kick
# off a five-minute network job at the moment of install.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.usstock.paper-v1"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
ACTION="${1:-status}"

case "$ACTION" in
install)
    mkdir -p "$HOME/Library/LaunchAgents"
    cat >"$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$REPO/scripts/paper_v1_daily.sh</string>
    </array>
    <key>WorkingDirectory</key><string>$REPO</string>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer></dict>
    </array>
    <key>RunAtLoad</key><false/>
    <key>StandardOutPath</key><string>$REPO/logs/paper_v1.launchd.log</string>
    <key>StandardErrorPath</key><string>$REPO/logs/paper_v1.launchd.log</string>
</dict>
</plist>
PLISTEOF
    mkdir -p "$REPO/logs"
    # `launchctl load` exits 0 in cases where nothing was registered, so trusting
    # its status printed "installed and loaded" over a schedule that did not
    # exist -- and the only symptom would have been a ledger that quietly stopped
    # accumulating. Prefer the modern subcommand, fall back to the old one, then
    # verify against launchctl list and refuse to claim success without it.
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
    launchctl unload "$PLIST" 2>/dev/null
    launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null \
        || launchctl load "$PLIST" 2>/dev/null

    if launchctl list | grep -qF "$LABEL"; then
        echo "已安裝並載入 $LABEL"
        echo "  排程：週一至週五 07:00（本地時間）"
        echo "  紀錄：logs/paper_v1.log"
        echo "  移除：bash scripts/install_paper_schedule.sh uninstall"
    else
        echo "寫入了 $PLIST，但 launchd 沒有註冊這個 agent。"
        echo "排程沒有生效——帳本不會自己推進。"
        echo "常見原因：這個指令是在沙箱或非登入 session 裡跑的，"
        echo "launchctl 連不到你的 GUI session。請在一般終端機裡直接跑："
        echo "  launchctl bootstrap gui/\$(id -u) $PLIST"
        exit 1
    fi
    ;;
uninstall)
    launchctl unload "$PLIST" 2>/dev/null
    rm -f "$PLIST"
    echo "已移除 $LABEL（帳本本身保留在 data/paper_v1/）"
    ;;
status)
    if [ -f "$PLIST" ]; then
        echo "plist 存在：$PLIST"
        launchctl list | grep -F "$LABEL" \
            && echo "狀態：已載入" \
            || echo "狀態：plist 在但未載入（跑 install 重新載入）"
    else
        echo "未安裝。跑：bash scripts/install_paper_schedule.sh install"
    fi
    ;;
*)
    echo "用法：bash scripts/install_paper_schedule.sh {install|status|uninstall}"
    exit 2
    ;;
esac
