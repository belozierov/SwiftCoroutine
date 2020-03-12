//
//  CoroutineError.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 11.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

public enum CoroutineError: Error {
    case mustBeCalledInsideCoroutine
    case wrongState
}
