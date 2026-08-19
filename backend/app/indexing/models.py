"""M2 索引事实数据的 SQLAlchemy 模型。"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    CHAR,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.sql import func
from sqlalchemy.dialects.mysql import INTEGER as MYSQL_INTEGER

from app.models.chat_history import Base

ASCII_SHA256 = CHAR(64, collation="ascii_bin")
ASCII_UUID = CHAR(36, collation="ascii_bin")
UNSIGNED_INTEGER = Integer().with_variant(MYSQL_INTEGER(unsigned=True), "mysql")
TABLE_OPTIONS = {
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_bin",
    "mysql_engine": "InnoDB",
}


class ContentBlob(Base):
    """由 SHA-256 唯一标识的不可变原始内容。"""

    __tablename__ = "content_blobs"
    __table_args__ = (
        CheckConstraint("byte_size >= 0", name="ck_content_blobs_byte_size"),
        TABLE_OPTIONS,
    )

    blob_id = Column(ASCII_SHA256, primary_key=True, comment="原始字节 SHA-256")
    byte_size = Column(BigInteger, nullable=False, comment="原始字节数")
    media_type = Column(String(255, collation="ascii_bin"), nullable=True, comment="MIME 类型")
    storage_uri = Column(String(1024, collation="utf8mb4_bin"), nullable=False, comment="持久内容 URI")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class IndexedDocument(Base):
    """独立于内容去重的用户逻辑文档。"""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "source_type",
            "display_name",
            name="uq_documents_owner_source_name",
        ),
        Index("ix_documents_user_id", "user_id"),
        TABLE_OPTIONS,
    )

    document_id = Column(ASCII_UUID, primary_key=True, comment="持久逻辑文档 UUID")
    user_id = Column(String(64, collation="ascii_bin"), nullable=False, comment="不透明认证主体 ID")
    source_type = Column(String(32, collation="ascii_bin"), nullable=False, comment="来源类型")
    display_name = Column(String(512, collation="utf8mb4_bin"), nullable=False, comment="用户可见名称")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class DocumentRevision(Base):
    """逻辑文档的一次不可变内容与提取文本快照。"""

    __tablename__ = "document_revisions"
    __table_args__ = (
        CheckConstraint("revision >= 1", name="ck_document_revisions_revision"),
        CheckConstraint(
            "legacy_index_only = 1 OR (blob_id IS NOT NULL AND "
            "normalized_text_sha256 IS NOT NULL AND normalized_text_uri IS NOT NULL)",
            name="ck_document_revisions_provenance",
        ),
        UniqueConstraint(
            "document_id",
            "revision",
            "legacy_index_only",
            name="uq_document_revisions_identity_legacy",
        ),
        TABLE_OPTIONS,
    )

    document_id = Column(
        ASCII_UUID,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        primary_key=True,
    )
    revision = Column(UNSIGNED_INTEGER, primary_key=True, comment="从 1 开始的单调版本")
    blob_id = Column(
        ASCII_SHA256,
        ForeignKey("content_blobs.blob_id", ondelete="RESTRICT"),
        nullable=True,
    )
    normalized_text_sha256 = Column(ASCII_SHA256, nullable=True, comment="规范化全文 SHA-256")
    normalized_text_uri = Column(
        String(1024, collation="utf8mb4_bin"),
        nullable=True,
        comment="规范化 UTF-8 文本的持久 URI",
    )
    legacy_index_only = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("0"),
        comment="历史索引缺少原件或区间",
    )
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class IndexVersion(Base):
    """规范索引配置及其确定性摘要。"""

    __tablename__ = "index_versions"
    __table_args__ = (
        CheckConstraint("contract_version = 1", name="ck_index_versions_contract_version"),
        TABLE_OPTIONS,
    )

    index_version = Column(ASCII_SHA256, primary_key=True, comment="规范配置 SHA-256")
    contract_version = Column(Integer, nullable=False, server_default=text("1"))
    config_json = Column(JSON, nullable=False, comment="已规范化的索引配置")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class IndexChunk(Base):
    """特定文档版本和索引版本下的稳定 chunk 来源锚点。"""

    __tablename__ = "index_chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "document_revision", "legacy_index_only"],
            [
                "document_revisions.document_id",
                "document_revisions.revision",
                "document_revisions.legacy_index_only",
            ],
            name="fk_index_chunks_document_revision",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "document_id",
            "document_revision",
            "index_version",
            "chunk_ordinal",
            name="uq_index_chunks_position",
        ),
        CheckConstraint("document_revision >= 1", name="ck_index_chunks_revision"),
        CheckConstraint("chunk_ordinal >= 0", name="ck_index_chunks_ordinal"),
        CheckConstraint("page_number IS NULL OR page_number >= 1", name="ck_index_chunks_page"),
        CheckConstraint(
            "(char_start IS NULL AND char_end IS NULL) OR "
            "(char_start IS NOT NULL AND char_end IS NOT NULL AND char_start >= 0 AND char_end > char_start)",
            name="ck_index_chunks_character_span",
        ),
        CheckConstraint(
            "legacy_index_only = 1 OR (char_start IS NOT NULL AND char_end IS NOT NULL)",
            name="ck_index_chunks_nonlegacy_span",
        ),
        Index("ix_index_chunks_document_revision", "document_id", "document_revision"),
        Index("ix_index_chunks_index_version", "index_version"),
        TABLE_OPTIONS,
    )

    chunk_id = Column(ASCII_SHA256, primary_key=True, comment="确定性 chunk SHA-256")
    document_id = Column(ASCII_UUID, nullable=False)
    document_revision = Column(UNSIGNED_INTEGER, nullable=False)
    index_version = Column(
        ASCII_SHA256,
        ForeignKey("index_versions.index_version", ondelete="RESTRICT"),
        nullable=False,
    )
    chunk_ordinal = Column(UNSIGNED_INTEGER, nullable=False, comment="从 0 开始的切片序号")
    content_sha256 = Column(ASCII_SHA256, nullable=False, comment="切片文本 UTF-8 SHA-256")
    page_number = Column(UNSIGNED_INTEGER, nullable=True, comment="1-based 页码")
    char_start = Column(UNSIGNED_INTEGER, nullable=True, comment="0-based 右开区间起点")
    char_end = Column(UNSIGNED_INTEGER, nullable=True, comment="0-based 右开区间终点")
    legacy_index_only = Column(Boolean, nullable=False, server_default=text("0"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
