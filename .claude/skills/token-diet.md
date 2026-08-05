---
name: token-diet
description: Token 节约策略 — 修改代码时只提交增量变更，避免全量重写浪费 token
---

# Token Diet Skill

当修改项目中的已有文件时，强制遵守以下规则以节约 token：

## 核心规则

### 1. Edit > Write
- **修改已有文件**：必须使用 `Edit` 工具（指定 `old_string` 和 `new_string`）
- **创建新文件**：才使用 `Write` 工具
- ❌ 禁止：用 Write 重写整个已有文件（这会把全文件内容计入 token）
- ✅ 正确：用 Edit 只替换变更的几行

### 2. 精准读取
- 读取大文件时使用 `offset` + `limit` 参数，只读需要的部分
- 先用 `Grep` 或 `Glob` 定位，再用 `Read` 精准读取
- ❌ 禁止：无脑 Read 整个 2000 行文件只为改其中 5 行
- ✅ 正确：Grep 定位行号 → Read(offset, limit=30) → Edit

### 3. 简洁回复
- 不要在回复中重复粘贴大段代码，用 `file_path:line_number` 引用
- 不要复述用户已经知道的信息
- 不要列出你不打算采用的方案

### 4. 精准 Git
- `git add` 只添加变更的文件，不要 `git add -A`
- commit message 简洁明了，不要写长篇大论

### 5. 并行操作
- 独立的文件修改可以并行执行（一次消息中多个 Edit 调用）
- 独立的搜索任务使用 Agent 并行派发
