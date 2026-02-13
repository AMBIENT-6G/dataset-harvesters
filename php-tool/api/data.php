<?php

declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');

require __DIR__ . '/db.php';

function json_error(string $message, int $code = 400): void
{
    http_response_code($code);
    echo json_encode(['error' => $message]);
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
        json_error('Failed to list tables: ' . mysqli_error($conn), 500);
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
        json_error('Failed to prepare column query: ' . mysqli_error($conn), 500);
    }
    mysqli_stmt_bind_param($stmt, 's', $table);
    if (!mysqli_stmt_execute($stmt)) {
        json_error('Failed to query columns: ' . mysqli_stmt_error($stmt), 500);
    }
    $result = mysqli_stmt_get_result($stmt);
    while ($row = mysqli_fetch_assoc($result)) {
        $columns[] = $row['column_name'];
    }
    mysqli_stmt_close($stmt);
    return $columns;
}

$selectedTables = filter_input(INPUT_POST, 'harvesters', FILTER_DEFAULT, FILTER_REQUIRE_ARRAY) ?? [];
$xAxis = trim((string) filter_input(INPUT_POST, 'x_axis', FILTER_DEFAULT));
$yAxis = trim((string) filter_input(INPUT_POST, 'y_axis', FILTER_DEFAULT));

$tables = fetch_tables($conn);
if (empty($tables)) {
    json_error('No harvester tables found.', 404);
}

$columns = fetch_columns($conn, $tables[0]);
if (empty($columns)) {
    json_error('No columns found for harvester tables.', 404);
}

$selectedTables = array_values(array_intersect($tables, $selectedTables));
if (empty($selectedTables)) {
    json_error('Select at least one harvester.');
}

if ($xAxis === '' || !in_array($xAxis, $columns, true)) {
    json_error('Select a valid X axis.');
}

if ($yAxis === '' || !in_array($yAxis, $columns, true)) {
    json_error('Select a valid Y axis.');
}

$safeX = '`' . str_replace('`', '``', $xAxis) . '`';
$safeY = '`' . str_replace('`', '``', $yAxis) . '`';

$selects = [];
foreach ($selectedTables as $table) {
    $safeTable = '`' . str_replace('`', '``', $table) . '`';
    $tableLiteral = mysqli_real_escape_string($conn, $table);
    $selects[] = "SELECT '$tableLiteral' AS harvester, $safeX AS x, $safeY AS y FROM $safeTable";
}

$sql = implode(' UNION ALL ', $selects);
$result = mysqli_query($conn, $sql);
if ($result === false) {
    json_error('Failed to query data: ' . mysqli_error($conn), 500);
}

$rows = [];
while ($row = mysqli_fetch_assoc($result)) {
    $rows[] = [
        'harvester' => $row['harvester'] ?? '',
        'x' => $row['x'],
        'y' => $row['y'],
    ];
}

mysqli_close($conn);

echo json_encode([
    'x_axis' => $xAxis,
    'y_axis' => $yAxis,
    'rows' => $rows,
]);

