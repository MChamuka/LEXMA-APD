import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';

class DatabaseService {
  static final DatabaseService instance = DatabaseService._init();
  static Database? _database;

  DatabaseService._init();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB('lexma_patients.db');
    return _database!;
  }

  Future<Database> _initDB(String filePath) async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, filePath);

    // Creates the database on the phone
    return await openDatabase(path, version: 1, onCreate: _createDB);
  }

  Future _createDB(Database db, int version) async {
    // Defines the exact structure of your saved medical records
    await db.execute('''
    CREATE TABLE assessments (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      face_prob REAL,
      spiral_prob REAL,
      voice_prob REAL,
      cdt_prob REAL,
      lifestyle_prob REAL,
      final_confidence REAL NOT NULL,
      diagnosis TEXT NOT NULL
    )
    ''');
  }

  Future<int> saveAssessment(Map<String, dynamic> row) async {
    final db = await instance.database;
    return await db.insert('assessments', row);
  }

  Future<List<Map<String, dynamic>>> fetchAllAssessments() async {
    final db = await instance.database;
    // Returns all past records, newest first
    return await db.query('assessments', orderBy: 'id DESC');
  }
}
