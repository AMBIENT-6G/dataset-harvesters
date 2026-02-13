<?php

declare(strict_types=1);

require __DIR__ . '/db.php';

function respond_error(string $message, int $code = 400): void
{
    http_response_code($code);
    header('Content-Type: text/plain; charset=utf-8');
    echo $message;
    exit;
}

function fetch_tables(mysqli $conn): array
{
    $tables = [];
    $sql = "SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND table_type = 'BASE TABLE'
        ORDER BY table_name";
    $result = mysqli_query($conn, $sql);
    if ($result === false) {
        respond_error('Failed to list tables: ' . mysqli_error($conn), 500);
    }
    while ($row = mysqli_fetch_assoc($result)) {
        $tables[] = $row['table_name'];
    }
    return $tables;
}

function fetch_columns(mysqli $conn, string $table): array
{
    $columns = [];
    $sql = "SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = ?
        ORDER BY ordinal_position";
    $stmt = mysqli_prepare($conn, $sql);
    if ($stmt === false) {
        respond_error('Failed to prepare column query: ' . mysqli_error($conn), 500);
    }
    mysqli_stmt_bind_param($stmt, 's', $table);
    if (!mysqli_stmt_execute($stmt)) {
        respond_error('Failed to query columns: ' . mysqli_stmt_error($stmt), 500);
    }
    $result = mysqli_stmt_get_result($stmt);
    while ($row = mysqli_fetch_assoc($result)) {
        $columns[] = $row['column_name'];
    }
    mysqli_stmt_close($stmt);
    return $columns;
}

$selectedTables = filter_input(INPUT_POST, 'harvesters', FILTER_DEFAULT, FILTER_REQUIRE_ARRAY) ?? [];
$selectedColumns = filter_input(INPUT_POST, 'columns', FILTER_DEFAULT, FILTER_REQUIRE_ARRAY) ?? [];
$includeHarvester = isset($_POST['include_harvester']);

$tables = fetch_tables($conn);
if (empty($tables)) {
    respond_error('No harvester tables found.', 404);
}

$columns = fetch_columns($conn, $tables[0]);
if (empty($columns)) {
    respond_error('No columns found for harvester tables.', 404);
}

$selectedTables = array_values(array_intersect($tables, $selectedTables));
$selectedColumns = array_values(array_intersect($columns, $selectedColumns));

if (empty($selectedTables)) {
    respond_error('Select at least one harvester.');
}
if (empty($selectedColumns)) {
    respond_error('Select at least one parameter.');
}

if (count($selectedTables) > 1) {
    $includeHarvester = true;
}

$orderedColumns = array_values(array_intersect($columns, $selectedColumns));
$header = $includeHarvester ? array_merge(['harvester'], $orderedColumns) : $orderedColumns;
$selects = [];

foreach ($selectedTables as $table) {
    $safeTable = '`' . str_replace('`', '``', $table) . '`';
    $colList = [];
    foreach ($orderedColumns as $col) {
        $colList[] = '`' . str_replace('`', '``', $col) . '`';
    }
    $cols = implode(', ', $colList);

    if ($includeHarvester) {
        $tableLiteral = mysqli_real_escape_string($conn, $table);
        $selects[] = "SELECT '$tableLiteral' AS harvester, $cols FROM $safeTable";
    } else {
        $selects[] = "SELECT $cols FROM $safeTable";
    }
}

$sql = implode(' UNION ALL ', $selects);
$result = mysqli_query($conn, $sql);
if ($result === false) {
    respond_error('Failed to query data: ' . mysqli_error($conn), 500);
}

$filename = 'harvester-data-' . date('Ymd-His') . '.csv';
header('Content-Type: text/csv; charset=utf-8');
header('Content-Disposition: attachment; filename="' . $filename . '"');

$out = fopen('php://output', 'w');
if ($out === false) {
    respond_error('Unable to open output stream.', 500);
}

fputcsv($out, $header);
while ($row = mysqli_fetch_assoc($result)) {
    $line = [];
    foreach ($header as $col) {
        $line[] = $row[$col] ?? '';
    }
    fputcsv($out, $line);
}

fclose($out);
mysqli_close($conn);

