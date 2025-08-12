#!/bin/bash

# OCR API Docker 部署管理腳本

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函數：打印彩色訊息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 函數：檢查 Docker 是否安裝
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安裝"
        echo "請先安裝 Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose 未安裝"
        echo "請先安裝 Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    print_info "Docker 環境檢查通過"
}

# 函數：檢查模型檔案
check_model() {
    if [ ! -f "best_lightweight_crnn_model.pth" ]; then
        print_error "找不到模型檔案: best_lightweight_crnn_model.pth"
        echo "請先訓練模型或複製模型檔案到當前目錄"
        exit 1
    fi
    print_info "模型檔案檢查通過"
}

# 函數：構建映像
build_image() {
    print_info "開始構建 Docker 映像..."
    docker-compose build --no-cache
    print_info "映像構建完成"
}

# 函數：啟動服務
start_service() {
    local scale=${1:-1}
    print_info "啟動服務 (實例數: $scale)..."
    
    if [ "$scale" -gt 1 ]; then
        docker-compose up -d --scale ocr-api=$scale
    else
        docker-compose up -d
    fi
    
    print_info "等待服務啟動..."
    sleep 5
    
    # 健康檢查
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_info "服務啟動成功！"
        
        # 顯示服務信息
        PUBLIC_IP=$(curl -s ifconfig.me)
        echo ""
        echo "=========================================="
        echo "OCR API 服務已啟動"
        echo "=========================================="
        echo "本地訪問: http://localhost:8000"
        echo "外網訪問: http://${PUBLIC_IP}:8000"
        echo "健康檢查: http://localhost:8000/health"
        echo "服務統計: http://localhost:8000/stats"
        echo "=========================================="
    else
        print_error "服務啟動失敗，請檢查日誌"
        docker-compose logs --tail=50
        exit 1
    fi
}

# 函數：停止服務
stop_service() {
    print_info "停止服務..."
    docker-compose down
    print_info "服務已停止"
}

# 函數：重啟服務
restart_service() {
    print_info "重啟服務..."
    docker-compose restart
    print_info "服務已重啟"
}

# 函數：查看日誌
view_logs() {
    local lines=${1:-100}
    docker-compose logs -f --tail=$lines ocr-api
}

# 函數：服務狀態
service_status() {
    echo "=========================================="
    echo "服務狀態"
    echo "=========================================="
    docker-compose ps
    echo ""
    echo "資源使用:"
    docker stats --no-stream ocr-api
    echo ""
    echo "健康狀態:"
    curl -s http://localhost:8000/health | python -m json.tool
    echo ""
    echo "服務統計:"
    curl -s http://localhost:8000/stats | python -m json.tool
}

# 函數：備份
backup() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    print_info "開始備份..."
    
    # 備份模型
    cp best_lightweight_crnn_model.pth "$backup_dir/"
    
    # 備份配置
    cp docker-compose.yml "$backup_dir/"
    cp Dockerfile "$backup_dir/"
    
    # 備份日誌
    docker-compose logs --no-color > "$backup_dir/logs.txt"
    
    # 打包
    tar -czf "$backup_dir.tar.gz" "$backup_dir"
    rm -rf "$backup_dir"
    
    print_info "備份完成: $backup_dir.tar.gz"
}

# 函數：清理
cleanup() {
    print_warn "清理 Docker 資源..."
    docker-compose down -v
    docker system prune -f
    print_info "清理完成"
}

# 函數：測試 API
test_api() {
    print_info "測試 API..."
    
    # 找一個測試圖片
    TEST_IMAGE=$(find captcha_auto_label -name "*.png" | head -1)
    
    if [ -z "$TEST_IMAGE" ]; then
        print_error "找不到測試圖片"
        return
    fi
    
    echo "使用測試圖片: $TEST_IMAGE"
    
    # 發送請求
    RESPONSE=$(curl -s -X POST -F "file=@${TEST_IMAGE}" http://localhost:8000/predict)
    
    echo "API 回應:"
    echo "$RESPONSE" | python -m json.tool
}

# 函數：性能測試
benchmark() {
    print_info "執行性能測試..."
    
    if [ -f "test_concurrent_client.py" ]; then
        python test_concurrent_client.py --benchmark
    else
        print_error "找不到測試腳本: test_concurrent_client.py"
    fi
}

# 主選單
show_menu() {
    echo ""
    echo "=========================================="
    echo "OCR API Docker 管理"
    echo "=========================================="
    echo "1. 構建映像"
    echo "2. 啟動服務（單實例）"
    echo "3. 啟動服務（多實例）"
    echo "4. 停止服務"
    echo "5. 重啟服務"
    echo "6. 查看日誌"
    echo "7. 服務狀態"
    echo "8. 測試 API"
    echo "9. 性能測試"
    echo "10. 備份"
    echo "11. 清理資源"
    echo "0. 退出"
    echo "=========================================="
}

# 主程式
main() {
    # 檢查環境
    check_docker
    check_model
    
    # 互動式選單
    while true; do
        show_menu
        read -p "請選擇操作 (0-11): " choice
        
        case $choice in
            1)
                build_image
                ;;
            2)
                start_service 1
                ;;
            3)
                read -p "請輸入實例數 (2-10): " instances
                start_service $instances
                ;;
            4)
                stop_service
                ;;
            5)
                restart_service
                ;;
            6)
                read -p "顯示最近幾行日誌 (預設 100): " lines
                lines=${lines:-100}
                view_logs $lines
                ;;
            7)
                service_status
                ;;
            8)
                test_api
                ;;
            9)
                benchmark
                ;;
            10)
                backup
                ;;
            11)
                read -p "確定要清理所有 Docker 資源嗎？(y/n): " confirm
                if [ "$confirm" = "y" ]; then
                    cleanup
                fi
                ;;
            0)
                print_info "退出"
                exit 0
                ;;
            *)
                print_error "無效的選擇"
                ;;
        esac
    done
}

# 支援命令列參數
if [ $# -gt 0 ]; then
    case "$1" in
        build)
            check_docker
            check_model
            build_image
            ;;
        start)
            check_docker
            check_model
            start_service ${2:-1}
            ;;
        stop)
            stop_service
            ;;
        restart)
            restart_service
            ;;
        logs)
            view_logs ${2:-100}
            ;;
        status)
            service_status
            ;;
        test)
            test_api
            ;;
        benchmark)
            benchmark
            ;;
        backup)
            backup
            ;;
        clean)
            cleanup
            ;;
        *)
            echo "用法: $0 [build|start|stop|restart|logs|status|test|benchmark|backup|clean]"
            echo "或直接執行 $0 進入互動式選單"
            exit 1
            ;;
    esac
else
    # 互動式模式
    main
fi