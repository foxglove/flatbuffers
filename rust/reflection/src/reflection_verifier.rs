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
use crate::{FlatbufferError, FlatbufferResult};
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

fn verify_struct(
    verifier: &mut Verifier,
    struct_object: &Object,
    struct_pos: usize,
    schema: &Schema,
    buf_loc_to_obj_idx: &mut HashMap<usize, i32>,
) -> FlatbufferResult<()> {
    verifier.range_in_buffer(struct_pos, struct_object.bytesize().try_into()?)?;
    for field in &struct_object.fields() {
        if field.type_().base_type() == BaseType::Obj {
            let obj_idx = field.type_().index();
            let child_obj = get_object(schema, obj_idx)?;
            if child_obj.is_struct() {
                let field_pos = struct_pos.saturating_add(field.offset().into());
                buf_loc_to_obj_idx.insert(field_pos, obj_idx);
                verify_struct(verifier, &child_obj, field_pos, schema, buf_loc_to_obj_idx)?;
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
                    let struct_size: usize = child_obj.bytesize().try_into()?;
                    let vector_size = vector_len.saturating_mul(struct_size.try_into()?);
                    verifier.range_in_buffer(vector_start, vector_size.try_into()?)?;
                    let vector_range = core::ops::Range {
                        start: vector_start,
                        end: vector_start.saturating_add(vector_size.try_into()?),
                    };
                    // A struct of size 0 holds no bytes to verify.
                    let step = struct_size.max(1);
                    for struct_pos in vector_range.step_by(step) {
                        buf_loc_to_obj_idx.insert(struct_pos, child_obj_idx);
                        verify_struct(
                            verifier,
                            &child_obj,
                            struct_pos,
                            schema,
                            buf_loc_to_obj_idx,
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
    let child_enum_idx = field_type.index();
    let child_enum = get_enum(schema, child_enum_idx)?;
    debug_assert!(!child_enum.values().is_empty());

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

        let enum_type = union_member_type(&child_enum, enum_value)?;
        let union_pos =
            union_offset_pos.saturating_add(verifier.get_uoffset(union_offset_pos)?.try_into()?);
        verifier.in_buffer::<u8>(union_pos)?;

        match enum_type.base_type() {
            BaseType::String => <&str>::run_verifier(verifier, union_pos)?,
            BaseType::Obj => {
                let child_obj = get_object(schema, enum_type.index())?;
                buf_loc_to_obj_idx.insert(union_pos, enum_type.index());
                if child_obj.is_struct() {
                    verify_struct(verifier, &child_obj, union_pos, schema, buf_loc_to_obj_idx)?
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

/// Returns the object at `index` in `schema`, or an error if there is none.
fn get_object<'a>(schema: &Schema<'a>, index: i32) -> FlatbufferResult<Object<'a>> {
    let objects = schema.objects();
    usize::try_from(index)
        .ok()
        .filter(|&i| i < objects.len())
        .map(|i| objects.get(i))
        .ok_or(FlatbufferError::InvalidSchema)
}

/// Returns the enum at `index` in `schema`, or an error if there is none.
fn get_enum<'a>(schema: &Schema<'a>, index: i32) -> FlatbufferResult<Enum<'a>> {
    let enums = schema.enums();
    usize::try_from(index)
        .ok()
        .filter(|&i| i < enums.len())
        .map(|i| enums.get(i))
        .ok_or(FlatbufferError::InvalidSchema)
}

/// Returns the type of the member of `union_enum` whose value is `tag`.
fn union_member_type<'a>(union_enum: &Enum<'a>, tag: u8) -> FlatbufferResult<Type<'a>> {
    union_enum
        .values()
        .lookup_by_key(i64::from(tag), |value, tag| {
            value.key_compare_with_value(*tag)
        })
        .and_then(|value| value.union_type())
        .ok_or(FlatbufferError::InvalidUnionEnum)
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
    if union_enum.values().is_empty() {
        return Err(FlatbufferError::InvalidUnionEnum);
    }

    let enum_offset = field
        .offset()
        .checked_sub(u16::try_from(SIZE_VOFFSET)?)
        .ok_or(FlatbufferError::InvalidUnionEnum)?;
    if let Some(enum_pos) = table_verifier.deref(enum_offset)? {
        let enum_value = table_verifier.verifier().get_u8(enum_pos)?;
        let enum_type = union_member_type(&union_enum, enum_value)?;

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
    use crate::test_util::{build_schema, field, vector_field, TestObject, TestUnion};

    fn le_u16s(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn verify(buffer: &[u8], schema: &[u8]) -> FlatbufferResult<()> {
        let schema = crate::reflection_generated::reflection::root_as_schema(schema).unwrap();
        verify_with_options(
            buffer,
            &schema,
            &VerifierOptions::default(),
            &mut HashMap::new(),
            Some("Root"),
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
            &[TestUnion::string()],
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

    /// Tables `A { x: uint }` and `B { v: [ubyte] }`, a table `Root` with `root_fields`, and a union
    /// `U` with `members`, where `A` is object 0 and `B` is object 1.
    fn union_schema(
        root_fields: Vec<crate::test_util::TestField>,
        members: Vec<(&'static str, BaseType, i32, i64)>,
    ) -> Vec<u8> {
        build_schema(
            &[
                TestObject::new_table("A", vec![field("x", BaseType::UInt, -1, 4)]),
                TestObject::new_table("B", vec![vector_field("v", BaseType::UByte, -1, 4)]),
                TestObject::new_table("Root", root_fields),
            ],
            &[TestUnion { name: "U", members }],
        )
    }

    /// `Root { u_type: U; u: U }`.
    fn single_union_schema(members: Vec<(&'static str, BaseType, i32, i64)>) -> Vec<u8> {
        union_schema(
            vec![
                field("u", BaseType::Union, 0, 6),
                field("u_type", BaseType::UType, 0, 4),
            ],
            members,
        )
    }

    /// A `Root` whose union field has tag `tag` and points at a table that is a valid `A` with
    /// `x = 4`. As a `B`, its vector would start past the end of the buffer.
    fn single_union_buffer(tag: u8) -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer.extend(12u32.to_le_bytes()); // root table offset
        buffer.extend(le_u16s(&[8, 12, 4, 8])); // root vtable
        buffer.extend(8i32.to_le_bytes()); // root table to vtable
        buffer.extend([tag, 0, 0, 0]); // `u_type` and padding
        buffer.extend(12u32.to_le_bytes()); // `u`
        buffer.extend(le_u16s(&[6, 8, 4])); // member table vtable
        buffer.extend([0, 0]); // padding
        buffer.extend(8i32.to_le_bytes()); // member table to vtable
        buffer.extend(4u32.to_le_bytes()); // `A.x`
        buffer
    }

    /// Union members `NONE = 0`, `B = 2` and `A = 3`, so that `B`'s value is `A`'s position.
    fn shuffled_members() -> Vec<(&'static str, BaseType, i32, i64)> {
        vec![
            ("NONE", BaseType::None, -1, 0),
            ("B", BaseType::Obj, 1, 2),
            ("A", BaseType::Obj, 0, 3),
        ]
    }

    #[test]
    fn test_union_member_is_verified_as_the_type_of_its_value() {
        let schema = single_union_schema(shuffled_members());
        // Tag 2 is `B`, so the member table is too short.
        assert!(verify(&single_union_buffer(2), &schema).is_err());
        // Tag 3 is `A`, which the member table is.
        verify(&single_union_buffer(3), &schema).unwrap();
    }

    #[test]
    fn test_union_with_non_contiguous_values_is_accepted() {
        let schema = single_union_schema(vec![
            ("NONE", BaseType::None, -1, 0),
            ("A", BaseType::Obj, 0, 5),
        ]);
        verify(&single_union_buffer(5), &schema).unwrap();
    }

    #[test]
    fn test_union_tag_without_member_is_rejected() {
        let schema = single_union_schema(shuffled_members());
        assert!(matches!(
            verify(&single_union_buffer(9), &schema),
            Err(FlatbufferError::InvalidUnionEnum)
        ));
    }

    #[test]
    fn test_vector_of_unions_member_is_verified_as_the_type_of_its_value() {
        let schema = union_schema(
            vec![
                vector_field("u", BaseType::Union, 0, 6),
                vector_field("u_type", BaseType::UType, 0, 4),
            ],
            shuffled_members(),
        );
        let buffer = |tag: u8| {
            let mut buffer = Vec::new();
            buffer.extend(12u32.to_le_bytes()); // root table offset
            buffer.extend(le_u16s(&[8, 12, 4, 8])); // root vtable
            buffer.extend(8i32.to_le_bytes()); // root table to vtable
            buffer.extend(8u32.to_le_bytes()); // `u_type`
            buffer.extend(12u32.to_le_bytes()); // `u`
            buffer.extend(1u32.to_le_bytes()); // `u_type` length
            buffer.extend([tag, 0, 0, 0]); // `u_type[0]` and padding
            buffer.extend(1u32.to_le_bytes()); // `u` length
            buffer.extend(12u32.to_le_bytes()); // `u[0]`
            buffer.extend(le_u16s(&[6, 8, 4])); // member table vtable
            buffer.extend([0, 0]); // padding
            buffer.extend(8i32.to_le_bytes()); // member table to vtable
            buffer.extend(4u32.to_le_bytes()); // `A.x`
            buffer
        };
        assert!(verify(&buffer(2), &schema).is_err());
        verify(&buffer(3), &schema).unwrap();
        assert!(matches!(
            verify(&buffer(9), &schema),
            Err(FlatbufferError::InvalidUnionEnum)
        ));
    }

    /// A `Root` table with one offset field (to `20`) at vtable offset 4.
    fn table_with_offset_field() -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer.extend(12u32.to_le_bytes()); // root table offset
        buffer.extend(le_u16s(&[6, 8, 4])); // root vtable
        buffer.extend([0, 0]); // padding
        buffer.extend(8i32.to_le_bytes()); // root table to vtable
        buffer.extend(4u32.to_le_bytes()); // offset field
        buffer.extend(0u32.to_le_bytes()); // target: an empty vector
        buffer
    }

    #[test]
    fn test_table_field_with_invalid_object_index_is_rejected() {
        let schema = build_schema(
            &[TestObject::new_table(
                "Root",
                vec![field("a", BaseType::Obj, 7, 4)],
            )],
            &[],
        );
        assert!(matches!(
            verify(&table_with_offset_field(), &schema),
            Err(FlatbufferError::InvalidSchema)
        ));
    }

    #[test]
    fn test_vector_field_with_invalid_object_index_is_rejected() {
        let schema = build_schema(
            &[TestObject::new_table(
                "Root",
                vec![vector_field("a", BaseType::Obj, 7, 4)],
            )],
            &[],
        );
        assert!(matches!(
            verify(&table_with_offset_field(), &schema),
            Err(FlatbufferError::InvalidSchema)
        ));
    }

    #[test]
    fn test_union_field_with_invalid_enum_index_is_rejected() {
        let schema = build_schema(
            &[
                TestObject::new_table("A", vec![field("x", BaseType::UInt, -1, 4)]),
                TestObject::new_table(
                    "Root",
                    vec![
                        field("u", BaseType::Union, 3, 6),
                        field("u_type", BaseType::UType, 3, 4),
                    ],
                ),
            ],
            &[],
        );
        assert!(matches!(
            verify(&single_union_buffer(1), &schema),
            Err(FlatbufferError::InvalidSchema)
        ));
    }

    #[test]
    fn test_union_field_without_room_for_its_type_field_is_rejected() {
        // A union field's type field is in the vtable slot before it, so a union field in the
        // first slot has none.
        let schema = build_schema(
            &[
                TestObject::new_table("A", vec![field("x", BaseType::UInt, -1, 4)]),
                TestObject::new_table("Root", vec![field("u", BaseType::Union, 0, 0)]),
            ],
            &[TestUnion {
                name: "U",
                members: vec![("NONE", BaseType::None, -1, 0), ("A", BaseType::Obj, 0, 1)],
            }],
        );
        assert!(verify(&single_union_buffer(1), &schema).is_err());
    }

    #[test]
    fn test_vector_of_empty_structs_is_accepted() {
        let schema = build_schema(
            &[
                TestObject::new_table("Root", vec![vector_field("s", BaseType::Obj, 1, 4)]),
                TestObject::new_struct("Z", 0, vec![]),
            ],
            &[],
        );
        let mut buffer = Vec::new();
        buffer.extend(12u32.to_le_bytes()); // root table offset
        buffer.extend(le_u16s(&[6, 8, 4])); // root vtable
        buffer.extend([0, 0]); // padding
        buffer.extend(8i32.to_le_bytes()); // root table to vtable
        buffer.extend(4u32.to_le_bytes()); // `s`
        buffer.extend(3u32.to_le_bytes()); // `s` length
        verify(&buffer, &schema).unwrap();
    }
}
