//! Builds reflection schemas and buffers for tests.

use crate::reflection_generated::reflection::{
    BaseType, Enum, EnumArgs, EnumVal, EnumValArgs, Field, FieldArgs, Object, ObjectArgs, Schema,
    SchemaArgs, Type, TypeArgs,
};
use flatbuffers::{FlatBufferBuilder, WIPOffset};

pub struct TestField {
    name: &'static str,
    base_type: BaseType,
    element: BaseType,
    index: i32,
    offset: u16,
    fixed_length: u16,
}

pub fn field(name: &'static str, base_type: BaseType, index: i32, offset: u16) -> TestField {
    TestField {
        name,
        base_type,
        element: BaseType::None,
        index,
        offset,
        fixed_length: 0,
    }
}

pub fn vector_field(name: &'static str, element: BaseType, index: i32, offset: u16) -> TestField {
    TestField {
        element,
        ..field(name, BaseType::Vector, index, offset)
    }
}

pub fn array_field(
    name: &'static str,
    element: BaseType,
    index: i32,
    offset: u16,
    fixed_length: u16,
) -> TestField {
    TestField {
        element,
        fixed_length,
        ..field(name, BaseType::Array, index, offset)
    }
}

pub struct TestObject {
    name: String,
    is_struct: bool,
    bytesize: i32,
    fields: Vec<TestField>,
}

impl TestObject {
    pub fn new_table(name: impl Into<String>, fields: Vec<TestField>) -> Self {
        Self {
            name: name.into(),
            is_struct: false,
            bytesize: 0,
            fields,
        }
    }

    pub fn new_struct(name: impl Into<String>, bytesize: i32, fields: Vec<TestField>) -> Self {
        Self {
            name: name.into(),
            is_struct: true,
            bytesize,
            fields,
        }
    }
}

/// A member of the union `U`: its name, and the base type it holds, if the schema gives one.
pub type TestUnionValue = (&'static str, Option<BaseType>);

/// The members of a union whose only member holds a string.
pub const STRING_UNION: &[TestUnionValue] = &[
    ("NONE", Some(BaseType::None)),
    ("Str", Some(BaseType::String)),
];

/// Builds a schema with `objects`, which must be sorted by name, with fields sorted by name. The
/// object named `Root`, if any, is the root table. If `union_values` is not empty, the schema has
/// a union enum `U` with those members, valued by position.
pub fn build_schema(objects: &[TestObject], union_values: &[TestUnionValue]) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::new();
    let mut object_offsets = Vec::new();
    let mut root_table = None;
    for object in objects {
        let mut field_offsets = Vec::new();
        for f in &object.fields {
            let field_name = builder.create_string(f.name);
            let type_ = Type::create(
                &mut builder,
                &TypeArgs {
                    base_type: f.base_type,
                    element: f.element,
                    index: f.index,
                    fixed_length: f.fixed_length,
                    ..Default::default()
                },
            );
            field_offsets.push(Field::create(
                &mut builder,
                &FieldArgs {
                    name: Some(field_name),
                    type_: Some(type_),
                    offset: f.offset,
                    ..Default::default()
                },
            ));
        }
        let name = builder.create_string(&object.name);
        let fields = builder.create_vector(&field_offsets);
        let offset = Object::create(
            &mut builder,
            &ObjectArgs {
                name: Some(name),
                fields: Some(fields),
                is_struct: object.is_struct,
                minalign: 1,
                bytesize: object.bytesize,
                ..Default::default()
            },
        );
        if object.name == "Root" {
            root_table = Some(offset);
        }
        object_offsets.push(offset);
    }
    let mut enum_offsets = Vec::new();
    if !union_values.is_empty() {
        let mut values = Vec::new();
        for (value, &(name, base_type)) in (0..).zip(union_values) {
            let name = builder.create_string(name);
            let union_type = base_type.map(|base_type| {
                Type::create(
                    &mut builder,
                    &TypeArgs {
                        base_type,
                        ..Default::default()
                    },
                )
            });
            values.push(EnumVal::create(
                &mut builder,
                &EnumValArgs {
                    name: Some(name),
                    value,
                    union_type,
                    ..Default::default()
                },
            ));
        }
        let name = builder.create_string("U");
        let values = builder.create_vector(&values);
        let underlying_type = Type::create(
            &mut builder,
            &TypeArgs {
                base_type: BaseType::UType,
                ..Default::default()
            },
        );
        enum_offsets.push(Enum::create(
            &mut builder,
            &EnumArgs {
                name: Some(name),
                values: Some(values),
                is_union: true,
                underlying_type: Some(underlying_type),
                ..Default::default()
            },
        ));
    }
    let objects = builder.create_vector(&object_offsets);
    let enums = builder.create_vector::<WIPOffset<Enum>>(&enum_offsets);
    let schema = Schema::create(
        &mut builder,
        &SchemaArgs {
            objects: Some(objects),
            enums: Some(enums),
            root_table,
            ..Default::default()
        },
    );
    builder.finish_minimal(schema);
    builder.finished_data().to_vec()
}

pub fn le_u16s(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A root table whose only field, at vtable offset 4 and buffer position 16, holds `field`.
pub fn one_field_table(field: &[u8]) -> Vec<u8> {
    let table_size = u16::try_from(4 + field.len()).unwrap();
    let mut buffer = Vec::new();
    buffer.extend(12u32.to_le_bytes()); // root table offset
    buffer.extend(le_u16s(&[6, table_size, 4])); // root vtable
    buffer.extend([0, 0]); // padding
    buffer.extend(8i32.to_le_bytes()); // root table to vtable
    buffer.extend(field);
    buffer
}
