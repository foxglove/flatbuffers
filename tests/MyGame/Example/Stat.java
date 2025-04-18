// automatically generated by the FlatBuffers compiler, do not modify

package MyGame.Example;

import com.google.flatbuffers.BaseVector;
import com.google.flatbuffers.BooleanVector;
import com.google.flatbuffers.ByteVector;
import com.google.flatbuffers.Constants;
import com.google.flatbuffers.DoubleVector;
import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.FloatVector;
import com.google.flatbuffers.IntVector;
import com.google.flatbuffers.LongVector;
import com.google.flatbuffers.ShortVector;
import com.google.flatbuffers.StringVector;
import com.google.flatbuffers.Struct;
import com.google.flatbuffers.Table;
import com.google.flatbuffers.UnionVector;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

@SuppressWarnings("unused")
public final class Stat extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_25_2_10(); }
  public static Stat getRootAsStat(ByteBuffer _bb) { return getRootAsStat(_bb, new Stat()); }
  public static Stat getRootAsStat(ByteBuffer _bb, Stat obj) { _bb.order(ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public Stat __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public String id() { int o = __offset(4); return o != 0 ? __string(o + bb_pos) : null; }
  public ByteBuffer idAsByteBuffer() { return __vector_as_bytebuffer(4, 1); }
  public ByteBuffer idInByteBuffer(ByteBuffer _bb) { return __vector_in_bytebuffer(_bb, 4, 1); }
  public long val() { int o = __offset(6); return o != 0 ? bb.getLong(o + bb_pos) : 0L; }
  public boolean mutateVal(long val) { int o = __offset(6); if (o != 0) { bb.putLong(o + bb_pos, val); return true; } else { return false; } }
  public int count() { int o = __offset(8); return o != 0 ? bb.getShort(o + bb_pos) & 0xFFFF : 0; }
  public boolean mutateCount(int count) { int o = __offset(8); if (o != 0) { bb.putShort(o + bb_pos, (short) count); return true; } else { return false; } }

  public static int createStat(FlatBufferBuilder builder,
      int idOffset,
      long val,
      int count) {
    builder.startTable(3);
    Stat.addVal(builder, val);
    Stat.addId(builder, idOffset);
    Stat.addCount(builder, count);
    return Stat.endStat(builder);
  }

  public static void startStat(FlatBufferBuilder builder) { builder.startTable(3); }
  public static void addId(FlatBufferBuilder builder, int idOffset) { builder.addOffset(0, idOffset, 0); }
  public static void addVal(FlatBufferBuilder builder, long val) { builder.addLong(1, val, 0L); }
  public static void addCount(FlatBufferBuilder builder, int count) { builder.addShort((short) count); builder.slot(2); }
  public static int endStat(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  @Override
  protected int keysCompare(Integer o1, Integer o2, ByteBuffer _bb) {
    int val_1 = _bb.getShort(__offset(8, o1, _bb)) & 0xFFFF;
    int val_2 = _bb.getShort(__offset(8, o2, _bb)) & 0xFFFF;
    return val_1 > val_2 ? 1 : val_1 < val_2 ? -1 : 0;
  }

  public static Stat __lookup_by_key(Stat obj, int vectorLocation, int key, ByteBuffer bb) {
    int span = bb.getInt(vectorLocation - 4);
    int start = 0;
    while (span != 0) {
      int middle = span / 2;
      int tableOffset = __indirect(vectorLocation + 4 * (start + middle), bb);
      int val = bb.getShort(__offset(8, bb.capacity() - tableOffset, bb)) & 0xFFFF;
      int comp = val > key ? 1 : val < key ? -1 : 0;
      if (comp > 0) {
        span = middle;
      } else if (comp < 0) {
        middle++;
        start += middle;
        span -= middle;
      } else {
        return (obj == null ? new Stat() : obj).__assign(tableOffset, bb);
      }
    }
    return null;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public Stat get(int j) { return get(new Stat(), j); }
    public Stat get(Stat obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
    public Stat getByKey(int key) {  return __lookup_by_key(null, __vector(), key, bb); }
    public Stat getByKey(Stat obj, int key) {  return __lookup_by_key(obj, __vector(), key, bb); }
  }
  public StatT unpack() {
    StatT _o = new StatT();
    unpackTo(_o);
    return _o;
  }
  public void unpackTo(StatT _o) {
    String _oId = id();
    _o.setId(_oId);
    long _oVal = val();
    _o.setVal(_oVal);
    int _oCount = count();
    _o.setCount(_oCount);
  }
  public static int pack(FlatBufferBuilder builder, StatT _o) {
    if (_o == null) return 0;
    int _id = _o.getId() == null ? 0 : builder.createString(_o.getId());
    return createStat(
      builder,
      _id,
      _o.getVal(),
      _o.getCount());
  }
}

