/*
 * Copyright 2018 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use crate::reflection_generated::reflection::{BaseType, Enum, Field, Object, Schema, Type};
use crate::{get_type_size, is_float, is_integer, FlatbufferError, FlatbufferResult};
use flatbuffers::{
    ForwardsUOffset, InvalidFlatbuffer, TableVerifier, UOffsetT, Vector, Verifiable, Verifier,
    VerifierOptions, SIZE_UOFFSET, SIZE_VOFFSET,
};
use std::collections::{HashMap, HashSet};

/// Verifies a buffer against its schema with custom verification options.
pub fn verify_with_options(
    buffer: &[u8],
    schema: &Schema,
    opts: &VerifierOptions,
    buf_loc_to_obj_idx: &mut HashMap<usize, i32>,
    root_table_name: Option<&str>,
) -> FlatbufferResult<()> {
    let mut verifier = Verifier::new(opts, buffer);
    let root_table = match root_table_name {
        Some(name) => schema
            .objects()
            .lookup_by_key(name, |o, k| o.key_compare_with_value(k)),
        None => schema.root_table(),
    };
    if let Some(table_object) = root_table {
        if let Ok(table_pos) = verifier.get_uoffset(0) {
            // Inserts -1 as object index for root table
            buf_loc_to_obj_idx.insert(table_pos.try_into()?, -1);
            let mut verified = HashSet::new();
            return verify_table(
                &mut verifier,
                &table_object,
                table_pos.try_into()?,
                schema,
                &mut verified,
                buf_loc_to_obj_idx,
            );
        }
    }
    Err(FlatbufferError::InvalidSchema)
}

/// Tables already verified, by buffer position and the location of their type in the schema.
type VerifiedTables = HashSet<(usize, usize)>;

/// Maximum depth of structs nested in structs.
const MAX_STRUCT_DEPTH: usize = 64;

fn get_object<'a>(schema: &Schema<'a>, index: i32) -> FlatbufferResult<Object<'a>> {
    let objects = schema.objects();
    usize::try_from(index)
        .ok()
        .filter(|&i| i < objects.len())
        .map(|i| objects.get(i))
        .ok_or(FlatbufferError::InvalidSchema)
}

fn get_enum<'a>(schema: &Schema<'a>, index: i32) -> FlatbufferResult<Enum<'a>> {
    let enums = schema.enums();
    usize::try_from(index)
        .ok()
        .filter(|&i| i < enums.len())
        .map(|i| enums.get(i))
        .ok_or(FlatbufferError::InvalidSchema)
}

/// Returns the type of the union member that the type byte `enum_value` selects.
fn get_union_member_type<'a>(union_enum: &Enum<'a>, enum_value: u8) -> FlatbufferResult<Type<'a>> {
    let values = union_enum.values();
    let index = usize::from(enum_value);
    if index >= values.len() {
        return Err(FlatbufferError::InvalidUnionEnum);
    }
    values
        .get(index)
        .union_type()
        .ok_or(FlatbufferError::InvalidUnionEnum)
}

fn struct_bytesize(struct_object: &Object) -> FlatbufferResult<usize> {
    usize::try_from(struct_object.bytesize())
        .ok()
        .filter(|&size| size > 0)
        .ok_or(FlatbufferError::InvalidSchema)
}

fn verify_table(
    verifier: &mut Verifier,
    table_object: &Object,
    table_pos: usize,
    schema: &Schema,
    verified: &mut VerifiedTables,
    buf_loc_to_obj_idx: &mut HashMap<usize, i32>,
) -> FlatbufferResult<()> {
    let key = (table_pos, table_object._tab.loc());
    if verified.contains(&key) {
        return Ok(());
    }

    let mut table_verifier = verifier.visit_table(table_pos)?;

    let fields = table_object.fields();
    for field in &fields {
        let field_name = field.name().to_owned();
        table_verifier = match field.type_().base_type() {
            BaseType::UType | BaseType::UByte => {
                table_verifier.visit_field::<u8>(field_name, field.offset(), field.required())?
            }
            BaseType::Bool => {
                table_verifier.visit_field::<bool>(field_name, field.offset(), field.required())?
            }
            BaseType::Byte => {
                table_verifier.visit_field::<i8>(field_name, field.offset(), field.required())?
            }
            BaseType::Short => {
                table_verifier.visit_field::<i16>(field_name, field.offset(), field.required())?
            }
            BaseType::UShort => {
                table_verifier.visit_field::<u16>(field_name, field.offset(), field.required())?
            }
            BaseType::Int => {
                table_verifier.visit_field::<i32>(field_name, field.offset(), field.required())?
            }
            BaseType::UInt => {
                table_verifier.visit_field::<u32>(field_name, field.offset(), field.required())?
            }
            BaseType::Long => {
                table_verifier.visit_field::<i64>(field_name, field.offset(), field.required())?
            }
            BaseType::ULong => {
                table_verifier.visit_field::<u64>(field_name, field.offset(), field.required())?
            }
            BaseType::Float => {
                table_verifier.visit_field::<f32>(field_name, field.offset(), field.required())?
            }
            BaseType::Double => {
                table_verifier.visit_field::<f64>(field_name, field.offset(), field.required())?
            }
            BaseType::String => table_verifier.visit_field::<ForwardsUOffset<&str>>(
                field_name,
                field.offset(),
                field.required(),
            )?,
            BaseType::Vector => {
                verify_vector(table_verifier, &field, schema, verified, buf_loc_to_obj_idx)?
            }
            BaseType::Obj => {
                if let Some(field_pos) = table_verifier.deref(field.offset())? {
                    let object_index = field.type_().index();
                    let child_obj = get_object(schema, object_index)?;
                    if child_obj.is_struct() {
                        buf_loc_to_obj_idx.insert(field_pos, object_index);
                        verify_struct(
                            table_verifier.verifier(),
                            &child_obj,
                            field_pos,
                            schema,
                            buf_loc_to_obj_idx,
                            0,
                        )?
                    } else {
                        let field_value = table_verifier.verifier().get_uoffset(field_pos)?;
                        let table_pos = field_pos.saturating_add(field_value.try_into()?);
                        buf_loc_to_obj_idx.insert(table_pos, object_index);
                        verify_table(
                            table_verifier.verifier(),
                            &child_obj,
                            table_pos,
                            schema,
                            verified,
                            buf_loc_to_obj_idx,
                        )?;
                    }
                } else if field.required() {
                    return InvalidFlatbuffer::new_missing_required(field.name().to_string())?;
                }
                table_verifier
            }
            BaseType::Union => {
                if let Some(field_pos) = table_verifier.deref(field.offset())? {
                    let field_value = table_verifier.verifier().get_uoffset(field_pos)?;
                    verify_union(
                        table_verifier,
                        &field,
                        field_pos.saturating_add(field_value.try_into()?),
                        schema,
                        verified,
                        buf_loc_to_obj_idx,
                    )?
                } else if field.required() {
                    return InvalidFlatbuffer::new_missing_required(field.name().to_string())?;
                } else {
                    table_verifier
                }
            }
            other => {
                return Err(FlatbufferError::UnsupportedTableFieldType(other));
            }
        };
    }

    table_verifier.finish();
    verified.insert(key);
    Ok(())
}

/// Verifies a struct `depth` levels below the outermost struct, and the structs nested in it.
///
/// Returns `InvalidSchema` unless every field is a scalar, a struct or a fixed-length array of
/// either, and ends inside the struct.
fn verify_struct(
    verifier: &mut Verifier,
    struct_object: &Object,
    struct_pos: usize,
    schema: &Schema,
    buf_loc_to_obj_idx: &mut HashMap<usize, i32>,
    depth: usize,
) -> FlatbufferResult<()> {
    if depth > MAX_STRUCT_DEPTH {
        return Err(FlatbufferError::InvalidSchema);
    }
    let bytesize = struct_bytesize(struct_object)?;
    verifier.range_in_buffer(struct_pos, bytesize)?;

    let mut fields_size = 0usize;
    for field in &struct_object.fields() {
        let type_ = field.type_();
        let (element, count) = match type_.base_type() {
            BaseType::Array => (type_.element(), usize::from(type_.fixed_length())),
            base_type => (base_type, 1),
        };
        let (element_size, child_obj) = if is_integer(element) || is_float(element) {
            (get_type_size(element), None)
        } else if element == BaseType::Obj {
            let child_obj = get_object(schema, type_.index())?;
            if !child_obj.is_struct() {
                return Err(FlatbufferError::InvalidSchema);
            }
            (struct_bytesize(&child_obj)?, Some(child_obj))
        } else {
            return Err(FlatbufferError::InvalidSchema);
        };

        let offset = usize::from(field.offset());
        let size = element_size
            .checked_mul(count)
            .ok_or(FlatbufferError::InvalidSchema)?;
        let end = offset
            .checked_add(size)
            .ok_or(FlatbufferError::InvalidSchema)?;
        // Capping the fields' total size at `bytesize` limits the structs nested at each depth,
        // so struct fields that overlap can't make verification exponential in the depth.
        fields_size = fields_size
            .checked_add(size)
            .ok_or(FlatbufferError::InvalidSchema)?;
        if end > bytesize || fields_size > bytesize {
            return Err(FlatbufferError::InvalidSchema);
        }

        if let Some(child_obj) = child_obj {
            for i in 0..count {
                // Doesn't overflow, since the field ends inside the struct and the struct inside
                // the buffer.
                let child_pos = struct_pos + offset + i * element_size;
                buf_loc_to_obj_idx.insert(child_pos, type_.index());
                verify_struct(
                    verifier,
                    &child_obj,
                    child_pos,
                    schema,
                    buf_loc_to_obj_idx,
                    depth + 1,
                )?;
            }
        }
    }
    Ok(())
}

fn verify_vector<'a, 'b, 'c>(
    mut table_verifier: TableVerifier<'a, 'b, 'c>,
    field: &Field,
    schema: &Schema,
    verified: &mut VerifiedTables,
    buf_loc_to_obj_idx: &mut HashMap<usize, i32>,
) -> FlatbufferResult<TableVerifier<'a, 'b, 'c>> {
    let field_name = field.name().to_owned();
    macro_rules! visit_vec {
        ($t:ty) => {
            table_verifier
                .visit_field::<ForwardsUOffset<Vector<$t>>>(
                    field_name,
                    field.offset(),
                    field.required(),
                )
                .map_err(FlatbufferError::VerificationError)
        };
    }
    match field.type_().element() {
        BaseType::UType | BaseType::UByte => visit_vec!(u8),
        BaseType::Bool => visit_vec!(bool),
        BaseType::Byte => visit_vec!(i8),
        BaseType::Short => visit_vec!(i16),
        BaseType::UShort => visit_vec!(u16),
        BaseType::Int => visit_vec!(i32),
        BaseType::UInt => visit_vec!(u32),
        BaseType::Long => visit_vec!(i64),
        BaseType::ULong => visit_vec!(u64),
        BaseType::Float => visit_vec!(f32),
        BaseType::Double => visit_vec!(f64),
        BaseType::String => visit_vec!(ForwardsUOffset<&str>),
        BaseType::Obj => {
            if let Some(field_pos) = table_verifier.deref(field.offset())? {
                let verifier = table_verifier.verifier();
                let vector_offset = verifier.get_uoffset(field_pos)?;
                let vector_pos = field_pos.saturating_add(vector_offset.try_into()?);
                let vector_len = verifier.get_uoffset(vector_pos)?;
                let vector_start = vector_pos.saturating_add(SIZE_UOFFSET);
                let child_obj_idx = field.type_().index();
                let child_obj = get_object(schema, child_obj_idx)?;
                if child_obj.is_struct() {
                    let struct_size = struct_bytesize(&child_obj)?;
                    let vector_size = vector_len.saturating_mul(struct_size.try_into()?);
                    verifier.range_in_buffer(vector_start, vector_size.try_into()?)?;
                    let vector_range = core::ops::Range {
                        start: vector_start,
                        end: vector_start.saturating_add(vector_size.try_into()?),
                    };
                    for struct_pos in vector_range.step_by(struct_size) {
                        buf_loc_to_obj_idx.insert(struct_pos, child_obj_idx);
                        verify_struct(
                            verifier,
                            &child_obj,
                            struct_pos,
                            schema,
                            buf_loc_to_obj_idx,
                            0,
                        )?;
                    }
                } else {
                    verifier.is_aligned::<UOffsetT>(vector_start)?;
                    let vector_size = vector_len.saturating_mul(SIZE_UOFFSET.try_into()?);
                    verifier.range_in_buffer(vector_start, vector_size.try_into()?)?;
                    let vector_range = core::ops::Range {
                        start: vector_start,
                        end: vector_start.saturating_add(vector_size.try_into()?),
                    };
                    for element_pos in vector_range.step_by(SIZE_UOFFSET) {
                        let table_pos = element_pos
                            .saturating_add(verifier.get_uoffset(element_pos)?.try_into()?);
                        buf_loc_to_obj_idx.insert(table_pos, child_obj_idx);
                        verify_table(
                            verifier,
                            &child_obj,
                            table_pos,
                            schema,
                            verified,
                            buf_loc_to_obj_idx,
                        )?;
                    }
                }
            } else if field.required() {
                return InvalidFlatbuffer::new_missing_required(field.name().to_string())?;
            }
            Ok(table_verifier)
        }
        BaseType::Union => {
            verify_vector_of_unions(table_verifier, field, schema, verified, buf_loc_to_obj_idx)
        }
        other => {
            return Err(FlatbufferError::UnsupportedVectorElementType(other));
        }
    }
}

fn verify_vector_of_unions<'a, 'b, 'c>(
    mut table_verifier: TableVerifier<'a, 'b, 'c>,
    field: &Field,
    schema: &Schema,
    verified: &mut VerifiedTables,
    buf_loc_to_obj_idx: &mut HashMap<usize, i32>,
) -> FlatbufferResult<TableVerifier<'a, 'b, 'c>> {
    let field_type = field.type_();
    // If the schema is valid, none of these asserts can fail.
    debug_assert_eq!(field_type.base_type(), BaseType::Vector);
    debug_assert_eq!(field_type.element(), BaseType::Union);
    let child_enum = get_enum(schema, field_type.index())?;

    // Assuming the schema is valid, the previous field must be the enum vector, which consists of
    // of 1-byte enums.
    let enum_field_offset = field
        .offset()
        .checked_sub(u16::try_from(SIZE_VOFFSET).unwrap())
        .ok_or(FlatbufferError::InvalidUnionEnum)?;

    // Either both vectors must be present, or both must be absent.
    let (value_field_pos, enum_field_pos) = match (
        table_verifier.deref(field.offset())?,
        table_verifier.deref(enum_field_offset)?,
    ) {
        (Some(value_field_pos), Some(enum_field_pos)) => (value_field_pos, enum_field_pos),
        (None, None) => {
            if field.required() {
                return InvalidFlatbuffer::new_missing_required(field.name().to_owned())?;
            } else {
                return Ok(table_verifier);
            }
        }
        _ => {
            return InvalidFlatbuffer::new_inconsistent_union(
                format!("{}_type", field.name()),
                field.name().to_owned(),
            )?;
        }
    };

    let verifier = table_verifier.verifier();
    let enum_vector_offset = verifier.get_uoffset(enum_field_pos)?;
    let enum_vector_pos = enum_field_pos.saturating_add(enum_vector_offset.try_into()?);
    let enum_vector_len = verifier.get_uoffset(enum_vector_pos)?;
    let enum_vector_start = enum_vector_pos.saturating_add(SIZE_UOFFSET);

    let value_vector_offset = verifier.get_uoffset(value_field_pos)?;
    let value_vector_pos = value_field_pos.saturating_add(value_vector_offset.try_into()?);
    let value_vector_len = verifier.get_uoffset(value_vector_pos)?;
    let value_vector_start = value_vector_pos.saturating_add(SIZE_UOFFSET);

    // Both vectors should have the same length.
    // The C++ verifier instead assumes that the length of the value vector is the length of the enum vector:
    // https://github.com/google/flatbuffers/blob/bd1b2d0bafb8be6059a29487db9e5ace5c32914d/src/reflection.cpp#L147-L162
    // This has been reported at https://github.com/google/flatbuffers/issues/8567
    if enum_vector_len != value_vector_len {
        return InvalidFlatbuffer::new_inconsistent_union(
            format!("{}_type", field.name()),
            field.name().to_owned(),
        )?;
    }

    // Regardless of its contents, the value vector in a vector of unions must be a vector of
    // offsets. Source: https://github.com/dvidelabs/flatcc/blob/master/doc/binary-format.md#unions
    verifier.is_aligned::<UOffsetT>(value_vector_start)?;
    let value_vector_size = value_vector_len.saturating_mul(SIZE_UOFFSET.try_into()?);
    verifier.range_in_buffer(value_vector_start, value_vector_size.try_into()?)?;
    let value_vector_range = core::ops::Range {
        start: value_vector_start,
        end: value_vector_start.saturating_add(value_vector_size.try_into()?),
    };

    // The enums must have a size of 1 byte, so we just use the length of the vector.
    let enum_vector_size = enum_vector_len;
    verifier.range_in_buffer(enum_vector_start, enum_vector_size.try_into()?)?;
    let enum_vector_range = core::ops::Range {
        start: enum_vector_start,
        end: enum_vector_start.saturating_add(enum_vector_size.try_into()?),
    };

    for (enum_pos, union_offset_pos) in
        enum_vector_range.zip(value_vector_range.step_by(SIZE_UOFFSET))
    {
        let enum_value = verifier.get_u8(enum_pos)?;
        if enum_value == 0 {
            // Discriminator is NONE. This should never happen: the C++ implementation forbids it.
            // For example, the C++ JSON parser forbids the type entry to be NONE for a vector of
            // unions: in
            // https://github.com/google/flatbuffers/blob/bd1b2d0bafb8be6059a29487db9e5ace5c32914d/src/idl_parser.cpp#L1383
            // the second argument is 'true', meaning that the NONE entry is skipped for the reverse
            // lookup.
            //
            // This should possibly be an error, but we can be forgiving and just ignore the
            // corresponding union entry entirely.
            continue;
        }

        let enum_type = get_union_member_type(&child_enum, enum_value)?;
        let union_pos =
            union_offset_pos.saturating_add(verifier.get_uoffset(union_offset_pos)?.try_into()?);
        verifier.in_buffer::<u8>(union_pos)?;

        match enum_type.base_type() {
            BaseType::String => <&str>::run_verifier(verifier, union_pos)?,
            BaseType::Obj => {
                let child_obj = get_object(schema, enum_type.index())?;
                buf_loc_to_obj_idx.insert(union_pos, enum_type.index());
                if child_obj.is_struct() {
                    verify_struct(
                        verifier,
                        &child_obj,
                        union_pos,
                        schema,
                        buf_loc_to_obj_idx,
                        0,
                    )?
                } else {
                    verify_table(
                        verifier,
                        &child_obj,
                        union_pos,
                        schema,
                        verified,
                        buf_loc_to_obj_idx,
                    )?;
                }
            }
            other => {
                return Err(FlatbufferError::UnsupportedUnionElementType(other));
            }
        }
    }
    Ok(table_verifier)
}

fn verify_union<'a, 'b, 'c>(
    mut table_verifier: TableVerifier<'a, 'b, 'c>,
    field: &Field,
    union_pos: usize,
    schema: &Schema,
    verified: &mut VerifiedTables,
    buf_loc_to_obj_idx: &mut HashMap<usize, i32>,
) -> FlatbufferResult<TableVerifier<'a, 'b, 'c>> {
    let union_enum = get_enum(schema, field.type_().index())?;
    let enum_offset = field
        .offset()
        .checked_sub(u16::try_from(SIZE_VOFFSET)?)
        .ok_or(FlatbufferError::InvalidUnionEnum)?;
    if let Some(enum_pos) = table_verifier.deref(enum_offset)? {
        let enum_value = table_verifier.verifier().get_u8(enum_pos)?;
        let enum_type = get_union_member_type(&union_enum, enum_value)?;

        match enum_type.base_type() {
            BaseType::String => <&str>::run_verifier(table_verifier.verifier(), union_pos)?,
            BaseType::Obj => {
                let child_obj = get_object(schema, enum_type.index())?;
                buf_loc_to_obj_idx.insert(union_pos, enum_type.index());
                if child_obj.is_struct() {
                    verify_struct(
                        table_verifier.verifier(),
                        &child_obj,
                        union_pos,
                        schema,
                        buf_loc_to_obj_idx,
                        0,
                    )?
                } else {
                    verify_table(
                        table_verifier.verifier(),
                        &child_obj,
                        union_pos,
                        schema,
                        verified,
                        buf_loc_to_obj_idx,
                    )?;
                }
            }
            other => {
                return Err(FlatbufferError::UnsupportedUnionElementType(other));
            }
        }
    } else {
        return InvalidFlatbuffer::new_inconsistent_union(
            format!("{}_type", field.name()),
            field.name().to_string(),
        )?;
    }

    Ok(table_verifier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reflection_generated::reflection::root_as_schema;
    use crate::test_util::{
        array_field, build_schema, field, vector_field, TestObject, STRING_UNION,
    };

    fn le_u16s(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn verify(buffer: &[u8], schema: &[u8]) -> FlatbufferResult<()> {
        let schema = root_as_schema(schema).unwrap();
        verify_with_options(
            buffer,
            &schema,
            &VerifierOptions::default(),
            &mut HashMap::new(),
            Some("Root"),
        )
    }

    /// A root table whose only field, at vtable offset 4 and buffer position 16, holds `field`.
    fn one_field_table(field: &[u8]) -> Vec<u8> {
        let table_size = u16::try_from(4 + field.len()).unwrap();
        let mut buffer = Vec::new();
        buffer.extend(12u32.to_le_bytes()); // root table offset
        buffer.extend(le_u16s(&[6, table_size, 4])); // root vtable
        buffer.extend([0, 0]); // padding
        buffer.extend(8i32.to_le_bytes()); // root table to vtable
        buffer.extend(field);
        buffer
    }

    /// A table `Root { a: U, a_type }`, where `a` is at vtable offset `a_offset` and both fields
    /// refer to the enum at `enum_index`. `U` holds a string.
    fn union_schema(enum_index: i32, a_offset: u16) -> Vec<u8> {
        build_schema(
            &[TestObject::new_table(
                "Root",
                vec![
                    field("a", BaseType::Union, enum_index, a_offset),
                    field("a_type", BaseType::UType, enum_index, 4),
                ],
            )],
            STRING_UNION,
        )
    }

    /// A `Root` for [union_schema] with `a_offset` 6, whose union holds the string "x" with type
    /// byte `a_type`.
    fn union_buffer(a_type: u8) -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer.extend(12u32.to_le_bytes()); // root table offset
        buffer.extend(le_u16s(&[8, 12, 4, 8])); // root vtable
        buffer.extend(8i32.to_le_bytes()); // root table to vtable
        buffer.extend([a_type, 0, 0, 0]); // `a_type` and padding
        buffer.extend(4u32.to_le_bytes()); // `a`
        buffer.extend(1u32.to_le_bytes()); // string length
        buffer.extend(b"x\0");
        buffer
    }

    /// A table `Root { s: S00 }`, where structs `S00` to `S{len - 1}` are each 1 byte. Each struct
    /// but the last holds the next in every field in `field_names`, all at offset 0. The last
    /// holds a `ubyte`.
    fn struct_chain_schema(len: usize, field_names: &[&'static str]) -> Vec<u8> {
        let mut objects = vec![TestObject::new_table(
            "Root",
            vec![field("s", BaseType::Obj, 1, 4)],
        )];
        for i in 0..len {
            let fields = if i + 1 < len {
                let next_index = i32::try_from(i + 2).unwrap();
                field_names
                    .iter()
                    .map(|&name| field(name, BaseType::Obj, next_index, 0))
                    .collect()
            } else {
                vec![field("x", BaseType::UByte, -1, 0)]
            };
            objects.push(TestObject::new_struct(format!("S{i:02}"), 1, fields));
        }
        build_schema(&objects, &[])
    }

    /// A table `Root { p: P }`, where 6-byte struct `P` holds `z: ubyte` at offset 0 and
    /// `arr: [C:arr_len]` at offset 2, and struct `C` is a 2-byte `ushort`.
    fn struct_array_schema(arr_len: u16) -> Vec<u8> {
        build_schema(
            &[
                TestObject::new_struct("C", 2, vec![field("x", BaseType::UShort, -1, 0)]),
                TestObject::new_struct(
                    "P",
                    6,
                    vec![
                        array_field("arr", BaseType::Obj, 0, 2, arr_len),
                        field("z", BaseType::UByte, -1, 0),
                    ],
                ),
                TestObject::new_table("Root", vec![field("p", BaseType::Obj, 1, 4)]),
            ],
            &[],
        )
    }

    #[test]
    fn test_table_shared_by_fields_of_different_types_is_verified_as_each_type() {
        let schema = build_schema(
            &[
                TestObject::new_table("A", vec![field("x", BaseType::UByte, -1, 4)]),
                TestObject::new_table(
                    "B",
                    vec![
                        field("q", BaseType::ULong, -1, 4),
                        field("y", BaseType::UByte, -1, 6),
                    ],
                ),
                TestObject::new_table(
                    "Root",
                    vec![
                        field("a", BaseType::Obj, 0, 4),
                        field("b", BaseType::Obj, 1, 6),
                    ],
                ),
            ],
            &[],
        );

        // `Root.a` and `Root.b` both point at one table, whose only data byte is the last byte of
        // the buffer. It is a valid `A` but too short for `B.q`.
        let mut buffer = Vec::new();
        buffer.extend(12u32.to_le_bytes()); // root table offset
        buffer.extend(le_u16s(&[8, 12, 4, 8])); // root vtable
        buffer.extend(8i32.to_le_bytes()); // root table to vtable
        buffer.extend(16u32.to_le_bytes()); // `a`
        buffer.extend(12u32.to_le_bytes()); // `b`
        buffer.extend(le_u16s(&[8, 5, 4, 4])); // shared table vtable
        buffer.extend(8i32.to_le_bytes()); // shared table to vtable
        buffer.push(1);

        assert!(verify(&buffer, &schema).is_err());
    }

    #[test]
    fn test_table_at_union_string_position_is_verified() {
        let schema = build_schema(
            &[
                TestObject::new_table("B", vec![field("q", BaseType::ULong, -1, 4)]),
                TestObject::new_table(
                    "Root",
                    vec![
                        field("a", BaseType::Union, 0, 6),
                        field("a_type", BaseType::UType, 0, 4),
                        field("t", BaseType::Obj, 0, 8),
                    ],
                ),
            ],
            STRING_UNION,
        );

        // Union `Root.a` holds a string, and `Root.t` points at the same position, which is not a
        // valid `B`.
        let mut buffer = Vec::new();
        buffer.extend(16u32.to_le_bytes()); // root table offset
        buffer.extend(le_u16s(&[10, 16, 4, 8, 12])); // root vtable
        buffer.extend([0, 0]); // padding
        buffer.extend(12i32.to_le_bytes()); // root table to vtable
        buffer.extend([1, 0, 0, 0]); // `a_type` and padding
        buffer.extend(8u32.to_le_bytes()); // `a`
        buffer.extend(4u32.to_le_bytes()); // `t`
        buffer.extend(1u32.to_le_bytes()); // string length
        buffer.extend(b"x\0");

        assert!(verify(&buffer, &schema).is_err());
    }

    #[test]
    fn test_table_shared_by_fields_of_same_type_is_accepted() {
        let schema = build_schema(
            &[
                TestObject::new_table("A", vec![field("x", BaseType::UByte, -1, 4)]),
                TestObject::new_table(
                    "Root",
                    vec![
                        field("a", BaseType::Obj, 0, 4),
                        field("b", BaseType::Obj, 0, 6),
                    ],
                ),
            ],
            &[],
        );

        let mut buffer = Vec::new();
        buffer.extend(12u32.to_le_bytes()); // root table offset
        buffer.extend(le_u16s(&[8, 12, 4, 8])); // root vtable
        buffer.extend(8i32.to_le_bytes()); // root table to vtable
        buffer.extend(16u32.to_le_bytes()); // `a`
        buffer.extend(12u32.to_le_bytes()); // `b`
        buffer.extend(le_u16s(&[8, 5, 4, 4])); // shared table vtable
        buffer.extend(8i32.to_le_bytes()); // shared table to vtable
        buffer.push(1);

        verify(&buffer, &schema).unwrap();
    }

    #[test]
    fn test_union_type_past_last_member_is_rejected() {
        let schema = union_schema(0, 6);

        // `U` has two members, so 1 is the last valid type byte.
        verify(&union_buffer(1), &schema).unwrap();
        assert_eq!(
            verify(&union_buffer(2), &schema),
            Err(FlatbufferError::InvalidUnionEnum)
        );
    }

    #[test]
    fn test_union_at_vtable_offset_0_is_rejected() {
        // The union's type field would be at vtable offset -2.
        let schema = union_schema(0, 0);

        assert_eq!(
            verify(&union_buffer(1), &schema),
            Err(FlatbufferError::InvalidUnionEnum)
        );
    }

    #[test]
    fn test_union_enum_index_past_last_enum_is_rejected() {
        let schema = union_schema(1, 6);

        assert_eq!(
            verify(&union_buffer(1), &schema),
            Err(FlatbufferError::InvalidSchema)
        );
    }

    #[test]
    fn test_object_index_past_last_object_is_rejected() {
        let schema = build_schema(
            &[TestObject::new_table(
                "Root",
                vec![field("t", BaseType::Obj, 1, 4)],
            )],
            &[],
        );

        assert_eq!(
            verify(&one_field_table(&4u32.to_le_bytes()), &schema),
            Err(FlatbufferError::InvalidSchema)
        );
    }

    #[test]
    fn test_vector_of_unions_member_without_type_is_rejected() {
        let schema = build_schema(
            &[TestObject::new_table(
                "Root",
                vec![
                    vector_field("v", BaseType::Union, 0, 6),
                    vector_field("v_type", BaseType::UType, 0, 4),
                ],
            )],
            &[("NONE", Some(BaseType::None)), ("A", None)],
        );

        // `v` holds one `A`.
        let mut buffer = Vec::new();
        buffer.extend(12u32.to_le_bytes()); // root table offset
        buffer.extend(le_u16s(&[8, 12, 4, 8])); // root vtable
        buffer.extend(8i32.to_le_bytes()); // root table to vtable
        buffer.extend(8u32.to_le_bytes()); // `v_type`
        buffer.extend(12u32.to_le_bytes()); // `v`
        buffer.extend(1u32.to_le_bytes()); // `v_type` length
        buffer.extend([1, 0, 0, 0]); // `v_type[0]` and padding
        buffer.extend(1u32.to_le_bytes()); // `v` length
        buffer.extend(4u32.to_le_bytes()); // `v[0]`
        buffer.extend(0u32.to_le_bytes()); // `v[0]`'s target

        assert_eq!(
            verify(&buffer, &schema),
            Err(FlatbufferError::InvalidUnionEnum)
        );
    }

    #[test]
    fn test_vector_of_zero_size_structs_is_rejected() {
        let schema = build_schema(
            &[
                TestObject::new_table("Root", vec![vector_field("v", BaseType::Obj, 1, 4)]),
                TestObject::new_struct("S", 0, vec![]),
            ],
            &[],
        );

        let mut buffer = one_field_table(&4u32.to_le_bytes()); // `v`
        buffer.extend(0u32.to_le_bytes()); // `v` length

        assert_eq!(
            verify(&buffer, &schema),
            Err(FlatbufferError::InvalidSchema)
        );
    }

    #[test]
    fn test_struct_containing_itself_is_rejected() {
        let schema = build_schema(
            &[
                TestObject::new_table("Root", vec![field("s", BaseType::Obj, 1, 4)]),
                TestObject::new_struct("S", 4, vec![field("s", BaseType::Obj, 1, 0)]),
            ],
            &[],
        );

        assert_eq!(
            verify(&one_field_table(&[0; 4]), &schema),
            Err(FlatbufferError::InvalidSchema)
        );
    }

    #[test]
    fn test_struct_with_out_of_line_field_is_rejected() {
        for (kind, s_field) in [
            ("string", field("f", BaseType::String, -1, 0)),
            ("vector", vector_field("f", BaseType::UByte, -1, 0)),
            ("table", field("f", BaseType::Obj, 2, 0)),
        ] {
            let schema = build_schema(
                &[
                    TestObject::new_table("Root", vec![field("s", BaseType::Obj, 1, 4)]),
                    TestObject::new_struct("S", 4, vec![s_field]),
                    TestObject::new_table("T", vec![]),
                ],
                &[],
            );

            assert_eq!(
                verify(&one_field_table(&[0; 4]), &schema),
                Err(FlatbufferError::InvalidSchema),
                "{kind} field in a struct"
            );
        }
    }

    #[test]
    fn test_struct_field_past_bytesize_is_rejected() {
        for (kind, s_field) in [
            ("ulong at offset 0", field("x", BaseType::ULong, -1, 0)),
            ("ubyte at offset 1", field("x", BaseType::UByte, -1, 1)),
        ] {
            let schema = build_schema(
                &[
                    TestObject::new_table("Root", vec![field("s", BaseType::Obj, 1, 4)]),
                    TestObject::new_struct("S", 1, vec![s_field]),
                ],
                &[],
            );

            // `s` is the last byte of the buffer.
            assert_eq!(
                verify(&one_field_table(&[7]), &schema),
                Err(FlatbufferError::InvalidSchema),
                "{kind} in a 1-byte struct"
            );
        }
    }

    #[test]
    fn test_overlapping_struct_fields_are_rejected() {
        // Each struct holds the next in both `a` and `b`, which overlap.
        let schema = struct_chain_schema(4, &["a", "b"]);

        assert_eq!(
            verify(&one_field_table(&[7]), &schema),
            Err(FlatbufferError::InvalidSchema)
        );
    }

    #[test]
    fn test_structs_nested_more_than_64_deep_are_rejected() {
        let buffer = one_field_table(&[7]);

        verify(&buffer, &struct_chain_schema(65, &["a"])).unwrap();
        assert_eq!(
            verify(&buffer, &struct_chain_schema(66, &["a"])),
            Err(FlatbufferError::InvalidSchema)
        );
    }

    #[test]
    fn test_struct_array_elements_are_recorded() {
        let schema_data = struct_array_schema(2);
        let schema = root_as_schema(&schema_data).unwrap();
        let buffer = one_field_table(&[9, 0, 1, 0, 2, 0]); // `p`

        let mut buf_loc_to_obj_idx = HashMap::new();
        verify_with_options(
            &buffer,
            &schema,
            &VerifierOptions::default(),
            &mut buf_loc_to_obj_idx,
            Some("Root"),
        )
        .unwrap();

        assert_eq!(
            buf_loc_to_obj_idx,
            HashMap::from([(12, -1), (16, 1), (18, 0), (20, 0)])
        );
    }

    #[test]
    fn test_struct_array_past_bytesize_is_rejected() {
        let schema = struct_array_schema(3);

        assert_eq!(
            verify(&one_field_table(&[9, 0, 1, 0, 2, 0]), &schema),
            Err(FlatbufferError::InvalidSchema)
        );
    }
}
