<?php

declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');

require __DIR__ . '/db.php';

$tables = [];
$columns = [];

$tableSql = "SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = DATABASE()
      AND table_type = 'BASE TABLE'
    ORDER BY table_name";

$tableResult = mysqli_query($conn, $tableSql);
if ($tableResult === false) {
    http_response_code(500);
    echo json_encode(['error' => 'Failed to list tables: ' . mysqli_error($conn)]);
    exit;
}

while ($row = mysqli_fetch_assoc($tableResult)) {
    $tables[] = $row['table_name'];
}

if (!empty($tables)) {
    $firstTable = $tables[0];
    $colSql = "SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = ?
        ORDER BY ordinal_position";
    $stmt = mysqli_prepare($conn, $colSql);
    if ($stmt === false) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to prepare column query: ' . mysqli_error($conn)]);
        exit;
    }
    mysqli_stmt_bind_param($stmt, 's', $firstTable);
    if (!mysqli_stmt_execute($stmt)) {
        http_response_code(500);
        echo json_encode(['error' => 'Failed to query columns: ' . mysqli_stmt_error($stmt)]);
        exit;
    }
    $result = mysqli_stmt_get_result($stmt);
    while ($row = mysqli_fetch_assoc($result)) {
        $columns[] = $row['column_name'];
    }
    mysqli_stmt_close($stmt);
}

mysqli_close($conn);

echo json_encode([
    'tables' => $tables,
    'columns' => $columns,
]);

