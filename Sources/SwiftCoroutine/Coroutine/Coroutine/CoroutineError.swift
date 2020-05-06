//
//  CoroutineError.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 05.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

public enum CoroutineError: Error {
    case canceled, calledOutsideCoroutine
}
