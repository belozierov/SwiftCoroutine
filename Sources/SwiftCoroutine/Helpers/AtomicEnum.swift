//
//  AtomicEnum.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 24.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

struct AtomicEnum<T: RawRepresentable> where T.RawValue == Int {
    
    private var _value: Int
    
    init(wrappedValue value: T) {
        _value = value.rawValue
    }
    
    var value: T {
        get { T(rawValue: _value)! }
        set { __atomicStore(pointer(), newValue.rawValue) }
    }
    
    mutating func update(_ newValue: T) -> T {
        T(rawValue: __atomicExchange(pointer(), newValue.rawValue))!
    }
    
    private mutating func pointer() -> OpaquePointer {
        OpaquePointer(UnsafeMutablePointer(&_value))
    }
    
}
