/*
 * Copyright 2025 Google Inc. All rights reserved.
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

use crate::r#struct::Struct;
use crate::reflection::BaseType;
use crate::reflection_generated::reflection::{Field, Object, Schema};
use crate::reflection_verifier::verify_with_options;
use crate::{
    get_any_field_float, get_any_field_float_in_struct, get_any_field_integer,
    get_any_field_integer_in_struct, get_any_field_string, get_any_field_string_in_struct,
    get_any_root, get_field_float, get_field_integer, get_field_string, get_field_struct,
    get_field_struct_in_struct, get_field_table, get_field_vector, get_type_size, is_float,
    is_integer, FlatbufferError, FlatbufferResult,
};
use flatbuffers::{EndianScalar, Follow, Table, Vector, VerifierOptions};
use num_traits::float::Float;
use num_traits::int::PrimInt;
use num_traits::FromPrimitive;
use std::any::type_name;
use std::collections::HashMap;

#[derive(Debug)]
pub struct SafeBuffer<'a> {
    buf: &'a [u8],
    schema: &'a Schema<'a>,
    root_table: Object<'a>,
}

impl<'a> SafeBuffer<'a> {
    pub fn new(buf: &'a [u8], schema: &'a Schema) -> FlatbufferResult<Self> {
        Self::new_with_options(buf, schema, &VerifierOptions::default())
    }

    pub fn new_with_options(
        buf: &'a [u8],
        schema: &'a Schema,
        opts: &VerifierOptions,
    ) -> FlatbufferResult<Self> {
        verify_with_options(&buf, schema, opts, &mut HashMap::new(), None)?;
        // Verification succeeds only if the schema has a root table.
        let root_table = schema.root_table().ok_or(FlatbufferError::InvalidSchema)?;
        Ok(SafeBuffer {
            buf,
            schema,
            root_table,
        })
    }

    /// Gets the root table in the buffer.
    pub fn get_root(&self) -> SafeTable {
        // SAFETY: the buffer was verified during construction.
        let table = unsafe { get_any_root(self.buf) };

        SafeTable {
            safe_buf: self,
            loc: table.loc(),
            obj: self.root_table,
        }
    }

    /// Returns the struct type that `field` holds, or an error if it does not hold a struct.
    fn struct_type(&self, field: &Field) -> FlatbufferResult<Object<'a>> {
        let type_ = field.type_();
        if type_.base_type() == BaseType::Obj {
            if let Some(obj) = usize::try_from(type_.index())
                .ok()
                .filter(|&i| i < self.schema.objects().len())
                .map(|i| self.schema.objects().get(i))
            {
                if obj.is_struct() {
                    return Ok(obj);
                }
            }
        }
        Err(type_mismatch("Struct", type_.base_type()))
    }
}

/// Returns the error for reading a field of type `field_type` as a `data_type`.
fn type_mismatch(data_type: &str, field_type: BaseType) -> FlatbufferError {
    FlatbufferError::FieldTypeMismatch(
        data_type.to_string(),
        field_type.variant_name().unwrap_or_default().to_string(),
    )
}

fn find_field_by_name<'a>(obj: &Object<'a>, field_name: &str) -> Option<Field<'a>> {
    obj.fields()
        .lookup_by_key(field_name, |field: &Field<'_>, key| {
            field.key_compare_with_value(key)
        })
}

#[derive(Debug)]
pub struct SafeTable<'a> {
    safe_buf: &'a SafeBuffer<'a>,
    loc: usize,
    obj: Object<'a>,
}

impl<'a> SafeTable<'a> {
    /// Gets an integer table field given its exact type. Returns default integer value if the field is not set. Returns [None] if no default value is found. Returns error if
    /// the table doesn't match the buffer or
    /// the [field_name] doesn't match the table or
    /// the field isn't an integer of `T`'s size.
    pub fn get_field_integer<T: for<'b> Follow<'b, Inner = T> + PrimInt + FromPrimitive>(
        &self,
        field_name: &str,
    ) -> FlatbufferResult<Option<T>> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            let base_type = field.type_().base_type();
            if !is_integer(base_type) || core::mem::size_of::<T>() != get_type_size(base_type) {
                return Err(type_mismatch(type_name::<T>(), base_type));
            }
            // SAFETY: the buffer was verified during construction, and `field` holds a `T`.
            Ok(
                unsafe {
                    get_field_integer::<T>(&Table::new(&self.safe_buf.buf, self.loc), &field)
                },
            )
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }

    /// Gets a floating point table field given its exact type. Returns default float value if the field is not set. Returns [None] if no default value is found. Returns error if
    /// the table doesn't match the buffer or
    /// the [field_name] doesn't match the table or
    /// the field isn't a float of `T`'s size.
    pub fn get_field_float<T: for<'b> Follow<'b, Inner = T> + Float>(
        &self,
        field_name: &str,
    ) -> FlatbufferResult<Option<T>> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            let base_type = field.type_().base_type();
            if !is_float(base_type) || core::mem::size_of::<T>() != get_type_size(base_type) {
                return Err(type_mismatch(type_name::<T>(), base_type));
            }
            // SAFETY: the buffer was verified during construction, and `field` holds a `T`.
            Ok(unsafe { get_field_float::<T>(&Table::new(&self.safe_buf.buf, self.loc), &field) })
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }

    /// Gets a String table field given its exact type. Returns empty string if the field is not set. Returns [None] if no default value is found. Returns error if
    /// the table doesn't match the buffer or
    /// the [field_name] doesn't match the table or
    /// the field isn't a string.
    pub fn get_field_string(&self, field_name: &str) -> FlatbufferResult<Option<&str>> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            let base_type = field.type_().base_type();
            if base_type != BaseType::String {
                return Err(type_mismatch("String", base_type));
            }
            // SAFETY: the buffer was verified during construction, and `field` holds a string.
            Ok(Some(unsafe {
                get_field_string(&Table::new(&self.safe_buf.buf, self.loc), &field)
            }))
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }

    /// Gets a [SafeStruct] table field given its exact type. Returns [None] if the field is not set. Returns error if
    /// the table doesn't match the buffer or
    /// the [field_name] doesn't match the table.
    pub fn get_field_struct(&self, field_name: &str) -> FlatbufferResult<Option<SafeStruct<'a>>> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            let obj = self.safe_buf.struct_type(&field)?;
            // SAFETY: the buffer was verified during construction, and `field` holds a struct.
            let optional_st =
                unsafe { get_field_struct(&Table::new(&self.safe_buf.buf, self.loc), &field) };
            Ok(optional_st.map(|st| SafeStruct {
                safe_buf: self.safe_buf,
                loc: st.loc(),
                obj,
            }))
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }

    /// Gets a Vector table field given its exact type. Returns empty vector if the field is not set. Returns error if
    /// the table doesn't match the buffer or
    /// the [field_name] doesn't match the table or
    /// the field isn't a vector of scalars of `T`'s size.
    ///
    /// Only the element size is checked against `T`, so for example `u32` reads a `[float]`
    /// vector's bits.
    ///
    /// `T` must be a scalar, so vectors of strings or tables can't be read:
    ///
    /// ```compile_fail,E0277
    /// # use flatbuffers::ForwardsUOffset;
    /// # fn get(table: &flatbuffers_reflection::safe_buffer::SafeTable) {
    /// table.get_field_vector::<ForwardsUOffset<&str>>("names");
    /// # }
    /// ```
    pub fn get_field_vector<T: Follow<'a> + EndianScalar>(
        &self,
        field_name: &str,
    ) -> FlatbufferResult<Vector<'a, T>> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            let type_ = field.type_();
            if type_.base_type() != BaseType::Vector {
                return Err(type_mismatch(type_name::<T>(), type_.base_type()));
            }
            let element = type_.element();
            if !(is_integer(element) || is_float(element))
                || core::mem::size_of::<T>() != get_type_size(element)
            {
                return Err(type_mismatch(type_name::<T>(), element));
            }
            // SAFETY: the buffer was verified during construction, `field` is a vector of scalars
            // the size of `T`, and `T` reads any bits as a valid value.
            Ok(unsafe { get_field_vector(&Table::new(&self.safe_buf.buf, self.loc), &field) })
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }

    /// Gets a [SafeTable] table field given its exact type. Returns [None] if the field is not set. Returns error if
    /// the table doesn't match the buffer or
    /// the [field_name] doesn't match the table.
    pub fn get_field_table(&self, field_name: &str) -> FlatbufferResult<Option<SafeTable<'a>>> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            let obj = (field.type_().base_type() == BaseType::Obj)
                .then(|| usize::try_from(field.type_().index()).ok())
                .flatten()
                .filter(|&i| i < self.safe_buf.schema.objects().len())
                .map(|i| self.safe_buf.schema.objects().get(i))
                .filter(|obj| !obj.is_struct());
            let Some(obj) = obj else {
                return Err(type_mismatch("Table", field.type_().base_type()));
            };

            // SAFETY: the buffer was verified during construction.
            let optional_table =
                unsafe { get_field_table(&Table::new(&self.safe_buf.buf, self.loc), &field) };
            Ok(optional_table.map(|t| SafeTable {
                safe_buf: self.safe_buf,
                loc: t.loc(),
                obj,
            }))
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }

    /// Returns the value of any table field as a 64-bit int, regardless of what type it is. Returns default integer if the field is not set or error if
    /// the value cannot be parsed as integer or
    /// the table doesn't match the buffer or
    /// the [field_name] doesn't match the table.
    /// [num_traits](https://docs.rs/num-traits/latest/num_traits/cast/trait.NumCast.html) is used for number casting.
    pub fn get_any_field_integer(&self, field_name: &str) -> FlatbufferResult<i64> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            // SAFETY: the buffer was verified during construction.
            unsafe { get_any_field_integer(&Table::new(&self.safe_buf.buf, self.loc), &field) }
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }

    /// Returns the value of any table field as a 64-bit floating point, regardless of what type it is. Returns default float if the field is not set or error if
    /// the value cannot be parsed as float or
    /// the table doesn't match the buffer or
    /// the [field_name] doesn't match the table.
    pub fn get_any_field_float(&self, field_name: &str) -> FlatbufferResult<f64> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            // SAFETY: the buffer was verified during construction.
            unsafe { get_any_field_float(&Table::new(&self.safe_buf.buf, self.loc), &field) }
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }

    /// Returns the string representation of any table field value (e.g. integer 123 is returned as "123"), regardless of what type it is. Returns empty string if the field is not set. Returns error if
    /// the table doesn't match the buffer or
    /// the [field_name] doesn't match the table.
    pub fn get_any_field_string(&self, field_name: &str) -> FlatbufferResult<String> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            // SAFETY: the buffer was verified during construction.
            unsafe {
                Ok(get_any_field_string(
                    &Table::new(&self.safe_buf.buf, self.loc),
                    &field,
                    self.safe_buf.schema,
                ))
            }
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }
}

#[derive(Debug)]
pub struct SafeStruct<'a> {
    safe_buf: &'a SafeBuffer<'a>,
    loc: usize,
    obj: Object<'a>,
}

impl<'a> SafeStruct<'a> {
    /// Gets a [SafeStruct] struct field given its exact type. Returns error if
    /// the struct doesn't match the buffer or
    /// the [field_name] doesn't match the struct.
    pub fn get_field_struct(&self, field_name: &str) -> FlatbufferResult<SafeStruct<'a>> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            let obj = self.safe_buf.struct_type(&field)?;
            // SAFETY: the buffer was verified during construction, and `field` holds a struct.
            let st = unsafe {
                get_field_struct_in_struct(&Struct::new(&self.safe_buf.buf, self.loc), &field)
            };
            Ok(SafeStruct {
                safe_buf: self.safe_buf,
                loc: st.loc(),
                obj,
            })
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }

    /// Returns the value of any struct field as a 64-bit int, regardless of what type it is. Returns error if
    /// the struct doesn't match the buffer or
    /// the [field_name] doesn't match the struct or
    /// the value cannot be parsed as integer.
    pub fn get_any_field_integer(&self, field_name: &str) -> FlatbufferResult<i64> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            // SAFETY: the buffer was verified during construction.
            unsafe {
                get_any_field_integer_in_struct(&Struct::new(&self.safe_buf.buf, self.loc), &field)
            }
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }

    /// Returns the value of any struct field as a 64-bit floating point, regardless of what type it is. Returns error if
    /// the struct doesn't match the buffer or
    /// the [field_name] doesn't match the struct or
    /// the value cannot be parsed as float.
    pub fn get_any_field_float(&self, field_name: &str) -> FlatbufferResult<f64> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            // SAFETY: the buffer was verified during construction.
            unsafe {
                get_any_field_float_in_struct(&Struct::new(&self.safe_buf.buf, self.loc), &field)
            }
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }

    /// Returns the string representation of any struct field value (e.g. integer 123 is returned as "123"), regardless of what type it is. Returns error if
    /// the struct doesn't match the buffer or
    /// the [field_name] doesn't match the struct.
    pub fn get_any_field_string(&self, field_name: &str) -> FlatbufferResult<String> {
        if let Some(field) = find_field_by_name(&self.obj, field_name) {
            // SAFETY: the buffer was verified during construction.
            unsafe {
                Ok(get_any_field_string_in_struct(
                    &Struct::new(&self.safe_buf.buf, self.loc),
                    &field,
                    self.safe_buf.schema,
                ))
            }
        } else {
            Err(FlatbufferError::FieldNotFound)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reflection_generated::reflection::root_as_schema;
    use crate::test_util::{build_schema, field, vector_field, TestObject};
    use flatbuffers::FlatBufferBuilder;

    /// A table `Root { f: float, s: string, v: [uint], w: [string], x: ushort }`.
    fn mixed_fields_schema() -> Vec<u8> {
        build_schema(
            &[TestObject::new_table(
                "Root",
                vec![
                    field("f", BaseType::Float, -1, 4),
                    field("s", BaseType::String, -1, 12),
                    vector_field("v", BaseType::UInt, -1, 6),
                    vector_field("w", BaseType::String, -1, 8),
                    field("x", BaseType::UShort, -1, 10),
                ],
            )],
            &[],
        )
    }

    fn mixed_fields_buffer() -> Vec<u8> {
        let mut builder = FlatBufferBuilder::new();
        let s = builder.create_string("text");
        let w_0 = builder.create_string("abc");
        let w = builder.create_vector(&[w_0]);
        let v = builder.create_vector(&[7u32]);
        let root = builder.start_table();
        builder.push_slot_always(4, 1.5f32);
        builder.push_slot_always(6, v);
        builder.push_slot_always(8, w);
        builder.push_slot_always(10, 3u16);
        builder.push_slot_always(12, s);
        let root = builder.end_table(root);
        builder.finish_minimal(root);
        builder.finished_data().to_vec()
    }

    fn mismatch(data_type: &str, field_type: &str) -> FlatbufferError {
        FlatbufferError::FieldTypeMismatch(data_type.to_string(), field_type.to_string())
    }

    #[test]
    fn test_get_field_integer_rejects_other_types_and_sizes() {
        let schema_data = mixed_fields_schema();
        let schema = root_as_schema(&schema_data).unwrap();
        let buffer = mixed_fields_buffer();
        let safe_buffer = SafeBuffer::new(&buffer, &schema).unwrap();
        let root = safe_buffer.get_root();

        assert_eq!(root.get_field_integer::<u16>("x"), Ok(Some(3)));
        assert_eq!(
            root.get_field_integer::<i64>("x"),
            Err(mismatch("i64", "UShort"))
        );
        assert_eq!(
            root.get_field_integer::<u32>("f"),
            Err(mismatch("u32", "Float"))
        );
    }

    #[test]
    fn test_get_field_float_rejects_other_types_and_sizes() {
        let schema_data = mixed_fields_schema();
        let schema = root_as_schema(&schema_data).unwrap();
        let buffer = mixed_fields_buffer();
        let safe_buffer = SafeBuffer::new(&buffer, &schema).unwrap();
        let root = safe_buffer.get_root();

        assert_eq!(root.get_field_float::<f32>("f"), Ok(Some(1.5)));
        assert_eq!(
            root.get_field_float::<f64>("f"),
            Err(mismatch("f64", "Float"))
        );
        assert_eq!(
            root.get_field_float::<f32>("v"),
            Err(mismatch("f32", "Vector"))
        );
    }

    #[test]
    fn test_get_field_string_rejects_other_types() {
        let schema_data = mixed_fields_schema();
        let schema = root_as_schema(&schema_data).unwrap();
        let buffer = mixed_fields_buffer();
        let safe_buffer = SafeBuffer::new(&buffer, &schema).unwrap();
        let root = safe_buffer.get_root();

        assert_eq!(root.get_field_string("s"), Ok(Some("text")));
        assert_eq!(root.get_field_string("f"), Err(mismatch("String", "Float")));
    }

    #[test]
    fn test_get_field_vector_rejects_other_types_and_element_types() {
        let schema_data = mixed_fields_schema();
        let schema = root_as_schema(&schema_data).unwrap();
        let buffer = mixed_fields_buffer();
        let safe_buffer = SafeBuffer::new(&buffer, &schema).unwrap();
        let root = safe_buffer.get_root();

        let v = root.get_field_vector::<u32>("v").unwrap();
        assert_eq!(v.iter().collect::<Vec<_>>(), [7]);
        assert_eq!(
            root.get_field_vector::<u16>("v").unwrap_err(),
            mismatch("u16", "UInt")
        );
        assert_eq!(
            root.get_field_vector::<u32>("w").unwrap_err(),
            mismatch("u32", "String")
        );
        assert_eq!(
            root.get_field_vector::<u8>("f").unwrap_err(),
            mismatch("u8", "Float")
        );
    }

    /// A table `Root { p: P }`, where struct `P { c: C, z: ubyte }` holds struct `C { x: uint }`
    /// at offset 0, so `P` and `C` start at the same position.
    fn nested_struct_schema() -> Vec<u8> {
        build_schema(
            &[
                TestObject::new_struct("C", 4, vec![field("x", BaseType::UInt, -1, 0)]),
                TestObject::new_struct(
                    "P",
                    8,
                    vec![
                        field("c", BaseType::Obj, 0, 0),
                        field("z", BaseType::UByte, -1, 4),
                    ],
                ),
                TestObject::new_table("Root", vec![field("p", BaseType::Obj, 1, 4)]),
            ],
            &[],
        )
    }

    fn nested_struct_buffer() -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer.extend(12u32.to_le_bytes()); // root table offset
        for x in [6u16, 12, 4] {
            buffer.extend(x.to_le_bytes()); // root vtable
        }
        buffer.extend([0, 0]); // padding
        buffer.extend(8i32.to_le_bytes()); // root table to vtable
        buffer.extend(7u32.to_le_bytes()); // `p.c.x`
        buffer.extend([9, 0, 0, 0]); // `p.z` and padding
        buffer
    }

    #[test]
    fn test_struct_and_nested_struct_at_same_position_keep_their_own_fields() {
        let schema_data = nested_struct_schema();
        let schema = root_as_schema(&schema_data).unwrap();
        let buffer = nested_struct_buffer();
        let safe_buffer = SafeBuffer::new(&buffer, &schema).unwrap();

        let p = safe_buffer
            .get_root()
            .get_field_struct("p")
            .unwrap()
            .unwrap();
        assert_eq!(p.get_any_field_integer("z").unwrap(), 9);
        let c = p.get_field_struct("c").unwrap();
        assert_eq!(c.get_any_field_integer("x").unwrap(), 7);
        assert!(c.get_any_field_integer("z").is_err());
    }

    #[test]
    fn test_get_field_struct_rejects_non_struct_field() {
        let schema_data = nested_struct_schema();
        let schema = root_as_schema(&schema_data).unwrap();
        let buffer = nested_struct_buffer();
        let safe_buffer = SafeBuffer::new(&buffer, &schema).unwrap();

        let p = safe_buffer
            .get_root()
            .get_field_struct("p")
            .unwrap()
            .unwrap();
        assert!(matches!(
            p.get_field_struct("z"),
            Err(FlatbufferError::FieldTypeMismatch(..))
        ));
    }
}
