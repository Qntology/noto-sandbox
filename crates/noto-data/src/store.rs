use anyhow::Result;
use lancedb::arrow::arrow_array::RecordBatch;
use lancedb::arrow::arrow_schema::SchemaRef;
use lancedb::{connect, Connection, Table};
use std::path::Path;

const LANCE_FORMAT_MARKER: &str = ".lancedb_format_version";
const LANCE_FORMAT_VALUE: &str = "0.41";

pub struct LanceStore {
    conn: Connection,
}

impl LanceStore {
    pub async fn open(uri: &str) -> Result<Self> {
        let is_local = !uri.contains("://");
        if is_local {
            let dir = Path::new(uri);
            let marker = dir.join(LANCE_FORMAT_MARKER);
            let is_legacy = dir.exists()
                && std::fs::read_to_string(&marker).ok().as_deref() != Some(LANCE_FORMAT_VALUE);
            if is_legacy {
                let probe = match connect(uri).execute().await {
                    Ok(c) => c.table_names().execute().await.is_ok(),
                    Err(_) => false,
                };
                if !probe {
                    let _ = std::fs::remove_dir_all(dir);
                }
            }
        }
        let conn = connect(uri).execute().await?;
        if is_local {
            let dir = Path::new(uri);
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join(LANCE_FORMAT_MARKER), LANCE_FORMAT_VALUE);
        }
        Ok(LanceStore { conn })
    }

    pub async fn ensure_table(&self, name: &str, schema: SchemaRef) -> Result<Table> {
        let existing = self.conn.table_names().execute().await?;
        if existing.iter().any(|n| n == name) {
            match self.conn.open_table(name).execute().await {
                Ok(t) => Ok(t),
                Err(_) => {
                    let _ = self.conn.drop_table(name, &[]).await;
                    Ok(self.conn.create_empty_table(name, schema).execute().await?)
                }
            }
        } else {
            Ok(self.conn.create_empty_table(name, schema).execute().await?)
        }
    }

    pub async fn append_all(&self, name: &str, batches: Vec<RecordBatch>) -> Result<()> {
        let batches: Vec<RecordBatch> = batches.into_iter().filter(|b| b.num_rows() > 0).collect();
        let Some(first) = batches.first() else {
            return Ok(());
        };
        let table = self.ensure_table(name, first.schema()).await?;
        table.add(batches).execute().await?;
        Ok(())
    }

    pub async fn replace_with(&self, name: &str, batches: Vec<RecordBatch>) -> Result<()> {
        let existing = self.conn.table_names().execute().await?;
        if existing.iter().any(|n| n == name) {
            self.conn.drop_table(name, &[]).await?;
        }
        self.append_all(name, batches).await
    }

    pub async fn append(&self, name: &str, batch: RecordBatch) -> Result<()> {
        self.append_all(name, vec![batch]).await
    }

    pub async fn row_count(&self, name: &str) -> Result<usize> {
        let existing = self.conn.table_names().execute().await?;
        if !existing.iter().any(|n| n == name) {
            return Ok(0);
        }
        let table = self.conn.open_table(name).execute().await?;
        Ok(table.count_rows(None).await?)
    }
}
