<?php

declare(strict_types=1);

// Fill in these constants in your environment or replace with literals.
if (!defined('MYSQL_SERVERNAME')) {
    define('MYSQL_SERVERNAME', '');
}
if (!defined('MYSQL_USERNAME')) {
    define('MYSQL_USERNAME', '');
}
if (!defined('MYSQL_PASSWORD')) {
    define('MYSQL_PASSWORD', '');
}
if (!defined('MYSQL_DBNAME')) {
    define('MYSQL_DBNAME', '');
}

$conn = mysqli_connect(MYSQL_SERVERNAME, MYSQL_USERNAME, MYSQL_PASSWORD, MYSQL_DBNAME);
if (!$conn) {
    http_response_code(500);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode(['error' => 'Database connection failed: ' . mysqli_connect_error()]);
    exit;
}

mysqli_set_charset($conn, 'utf8mb4');

